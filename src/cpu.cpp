#define NOMINMAX

#include "raster_core.h"
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <thread>
#include <omp.h>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <fstream>
#include <cassert>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <curl/curl.h>
#include <zlib.h>
#include <zstd.h>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include "gdal_priv.h"
#include "cpl_string.h"
#include "cpl_progress.h"
#include <pybind11/pybind11.h>

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t c, const char* f, int l, bool a = true) {
    if (c != cudaSuccess) { fprintf(stderr,"GPU Error: %s at %s:%d\n",cudaGetErrorString(c),f,l); if(a)exit(c); }
}

static const int kNumThreads = std::max(1,(int)std::thread::hardware_concurrency()-1);

enum TokenType { NUMBER, OPERATOR, PARENTHESIS, BAND, COMPARE_OP, LOGICAL_OP, FUNC_KW, ARG_SEP };
struct Token { TokenType type; std::string value; };

struct GDALDatasetContainer {
    std::vector<GDALDataset*> in_datasets;
    std::vector<std::vector<GDALRasterBand*>> bands;
    GDALDatasetContainer(int nt, int nb) {
        in_datasets.resize(nt,nullptr); bands.resize(nt);
        for(int i=0;i<nt;i++) bands[i].resize(nb,nullptr);
    }
};

static size_t get_avail_ram() {
    std::ifstream m("/proc/meminfo"); std::string line; size_t kb=0;
    while(std::getline(m,line))
        if(line.find("MemAvailable:")==0){sscanf(line.c_str(),"MemAvailable: %zu kB",&kb);break;}
    return kb*1024;
}

// ───────────────────────── TOKENIZER & COMPILER ──────────────────────────────

static int prec(const std::string& op){ return (op=="+"||op=="-")?1:(op=="*"||op=="/")?2:0; }

static std::vector<Token> tokenize_rpn(const std::string& expr, std::set<int>& bi){
    std::vector<Token> toks,ops,out;
    for(int i=0;i<(int)expr.size();i++){
        char c=expr[i]; if(c==' ')continue;
        if(c=='('||c==')')toks.push_back({PARENTHESIS,{c}});
        else if(c=='+'||c=='-'||c=='*'||c=='/')toks.push_back({OPERATOR,{c}});
        else if(c=='B'||c=='b'){
            i++;std::string n; while(i<(int)expr.size()&&isdigit(expr[i]))n+=expr[i++];
            bi.insert(std::stoi(n)-1); toks.push_back({BAND,std::to_string(std::stoi(n)-1)}); i--;
        } else if(isdigit(c)||c=='.'){
            std::string n; while(i<(int)expr.size()&&(isdigit(expr[i])||expr[i]=='.'))n+=expr[i++];
            toks.push_back({NUMBER,n}); i--;
        }
    }
    for(auto& t:toks){
        if(t.type==BAND||t.type==NUMBER){out.push_back(t);}
        else if(t.type==PARENTHESIS){
            if(t.value=="(")ops.push_back(t);
            else{while(!ops.empty()&&ops.back().value!="("){out.push_back(ops.back());ops.pop_back();}if(!ops.empty())ops.pop_back();}
        } else {
            while(!ops.empty()&&ops.back().value!="("&&prec(ops.back().value)>=prec(t.value)){out.push_back(ops.back());ops.pop_back();}
            ops.push_back(t);
        }
    }
    while(!ops.empty()){out.push_back(ops.back());ops.pop_back();}
    return out;
}

static std::vector<Instruction> compile_rpn(const std::vector<Token>& rpn, const std::set<int>& bi){
    std::vector<Instruction> insts;
    for(auto& t:rpn){
        Instruction ins{};
        if(t.type==NUMBER){ins.op=OP_LOAD_CONST;ins.constant=std::stof(t.value);ins.band_index=-1;}
        else if(t.type==BAND){
            ins.op=OP_LOAD_BAND;ins.constant=0.f;
            ins.band_index=(int)std::distance(bi.begin(),bi.find(std::stoi(t.value)));
        } else {
            switch(t.value[0]){case '+':ins.op=OP_ADD;break;case '-':ins.op=OP_SUB;break;
                                case '*':ins.op=OP_MUL;break;case '/':ins.op=OP_DIV;break;default:ins.op=OP_ADD;}
        }
        insts.push_back(ins);
    }
    return insts;
}

static std::string str_toupper(std::string s){ for(auto& c:s) c=(char)toupper((unsigned char)c); return s; }

static std::vector<Token> lex_reclass(const std::string& expr, std::set<int>& bi){
    std::vector<Token> toks; int n=(int)expr.size();
    for(int i=0;i<n;){
        char c=expr[i];
        if(c==' '||c=='\t'){i++;continue;}
        if(c==','){ toks.push_back({ARG_SEP,","}); i++; continue; }
        if(c=='('||c==')'){ toks.push_back({PARENTHESIS,{c}}); i++; continue; }
        if(c=='+'||c=='-'||c=='*'||c=='/'){ toks.push_back({OPERATOR,{c}}); i++; continue; }
        if(c=='>'||c=='<'||c=='='||c=='!'){
            std::string op{c}; i++;
            if(i<n&&expr[i]=='='){op+=expr[i++];}
            toks.push_back({COMPARE_OP,op}); continue;
        }
        if(isdigit(c)||c=='.'){
            std::string num; while(i<n&&(isdigit(expr[i])||expr[i]=='.')) num+=expr[i++];
            toks.push_back({NUMBER,num}); continue;
        }
        if(isalpha(c)||c=='_'){
            std::string word; while(i<n&&(isalnum(expr[i])||expr[i]=='_')) word+=expr[i++];
            if((word[0]=='B'||word[0]=='b')&&word.size()>1&&isdigit(word[1])){
                int idx=std::stoi(word.substr(1))-1; bi.insert(idx);
                toks.push_back({BAND,std::to_string(idx)}); continue;
            }
            std::string up=str_toupper(word);
            if(up=="AND"||up=="OR")  { toks.push_back({LOGICAL_OP,up}); continue; }
            if(up=="NOT")            { toks.push_back({LOGICAL_OP,"NOT"}); continue; }
            if(up=="IF"||up=="BETWEEN"||up=="CLAMP"||up=="MIN"||up=="MAX"){ toks.push_back({FUNC_KW,up}); continue; }
            try { std::stof(word); toks.push_back({NUMBER,word}); } catch(...) {}
            continue;
        }
        i++; 
    }
    return toks;
}

static int prec_reclass(const std::string& op){
    if(op=="*"||op=="/") return 5;
    if(op=="+"||op=="-") return 4;
    if(op==">"||op=="<"||op==">="||op=="<="||op=="=="||op=="!=") return 3;
    if(op=="NOT") return 2; 
    if(op=="AND") return 1;
    if(op=="OR")  return 0;
    return -1;
}
static bool is_right_assoc(const std::string& op){ return op=="NOT"; }

static std::vector<Token> shunting_yard_reclass(const std::vector<Token>& toks){
    std::vector<Token> out, ops;
    for(auto& t:toks){
        if(t.type==NUMBER||t.type==BAND){ out.push_back(t); } 
        else if(t.type==FUNC_KW){ ops.push_back(t); } 
        else if(t.type==PARENTHESIS&&t.value=="("){ ops.push_back(t); } 
        else if(t.type==ARG_SEP){
            while(!ops.empty()&&!(ops.back().type==PARENTHESIS&&ops.back().value=="(")){ out.push_back(ops.back()); ops.pop_back(); }
        } else if(t.type==PARENTHESIS&&t.value==")"){
            while(!ops.empty()&&!(ops.back().type==PARENTHESIS&&ops.back().value=="(")){ out.push_back(ops.back()); ops.pop_back(); }
            if(!ops.empty()) ops.pop_back(); 
            if(!ops.empty()&&ops.back().type==FUNC_KW){ out.push_back(ops.back()); ops.pop_back(); }
        } else if(t.type==OPERATOR||t.type==COMPARE_OP||t.type==LOGICAL_OP){
            int tp=prec_reclass(t.value); bool rta=is_right_assoc(t.value);
            while(!ops.empty()){
                auto& top=ops.back();
                if(top.type==PARENTHESIS||top.type==FUNC_KW) break;
                int topP=prec_reclass(top.value); if(topP<0) break;
                if(rta?topP>tp:topP>=tp){ out.push_back(top); ops.pop_back(); } else break;
            }
            ops.push_back(t);
        }
    }
    while(!ops.empty()){ out.push_back(ops.back()); ops.pop_back(); }
    return out;
}

static std::vector<Token> tokenize_reclass(const std::string& expr, std::set<int>& bi){ return shunting_yard_reclass(lex_reclass(expr,bi)); }

static std::vector<Instruction> compile_reclass(const std::vector<Token>& rpn, const std::set<int>& bi){
    std::vector<Instruction> insts;
    for(auto& t:rpn){
        Instruction ins{}; ins.constant=0.f; ins.band_index=-1;
        if(t.type==NUMBER){ ins.op=OP_LOAD_CONST; ins.constant=std::stof(t.value); } 
        else if(t.type==BAND){ ins.op=OP_LOAD_BAND; ins.band_index=(int)std::distance(bi.begin(),bi.find(std::stoi(t.value))); } 
        else if(t.type==OPERATOR){ switch(t.value[0]){case '+':ins.op=OP_ADD;break;case '-':ins.op=OP_SUB;break;case '*':ins.op=OP_MUL;break;case '/':ins.op=OP_DIV;break;default:ins.op=OP_ADD;} } 
        else if(t.type==COMPARE_OP){
            if(t.value==">") ins.op=OP_GT; else if(t.value=="<") ins.op=OP_LT; else if(t.value==">=") ins.op=OP_GTE; else if(t.value=="<=") ins.op=OP_LTE; else if(t.value=="==") ins.op=OP_EQ; else if(t.value=="!=") ins.op=OP_NEQ;
        } else if(t.type==LOGICAL_OP){
            if(t.value=="AND") ins.op=OP_AND; else if(t.value=="OR") ins.op=OP_OR; else if(t.value=="NOT") ins.op=OP_NOT;
        } else if(t.type==FUNC_KW){
            if(t.value=="IF") ins.op=OP_IF; else if(t.value=="BETWEEN") ins.op=OP_BETWEEN; else if(t.value=="CLAMP") ins.op=OP_CLAMP; else if(t.value=="MIN") ins.op=OP_MIN2; else if(t.value=="MAX") ins.op=OP_MAX2;
        }
        insts.push_back(ins);
    }
    return insts;
}

// ───────────────────────── THREAD BUFFERS ────────────────────────────────────

struct ThreadBufs {
    float* h_master=nullptr;
    std::vector<float*> h_bands,d_bands;
    float* h_out=nullptr; float* d_out=nullptr;
    float** d_band_ptrs=nullptr;
    cudaStream_t stream{};
    void alloc(size_t band_bytes,int nb){
        h_bands.resize(nb); d_bands.resize(nb);
        cudaCheck(cudaHostAlloc(&h_master,band_bytes*nb,cudaHostAllocMapped|cudaHostAllocWriteCombined));
        memset(h_master,0,band_bytes*nb);
        for(int i=0;i<nb;i++){
            float* h=h_master+i*(band_bytes/sizeof(float)); float* d;
            cudaCheck(cudaHostGetDevicePointer((void**)&d,h,0));
            h_bands[i]=h; d_bands[i]=d;
        }
        cudaCheck(cudaHostAlloc(&h_out,band_bytes,cudaHostAllocMapped));
        cudaCheck(cudaHostGetDevicePointer((void**)&d_out,h_out,0));
        memset(h_out,0,band_bytes);
        cudaCheck(cudaMalloc(&d_band_ptrs,nb*sizeof(float*)));
        cudaCheck(cudaMemcpy(d_band_ptrs,d_bands.data(),nb*sizeof(float*),cudaMemcpyHostToDevice));
        cudaCheck(cudaStreamCreate(&stream));
    }
    void free_all(){cudaFreeHost(h_master);cudaFreeHost(h_out);cudaFree(d_band_ptrs);cudaStreamDestroy(stream);}
};

// ───────────────────────── S3 / SIGV4 ────────────────────────────────────────

static std::string hex_enc(const uint8_t* d,size_t n){
    static const char h[]="0123456789abcdef"; std::string r(n*2,0);
    for(size_t i=0;i<n;i++){r[i*2]=h[d[i]>>4];r[i*2+1]=h[d[i]&0xf];} return r;
}
static std::string sha256h(const std::string& s){uint8_t h[32];SHA256((const uint8_t*)s.data(),s.size(),h);return hex_enc(h,32);}
static std::vector<uint8_t> hmacv(const std::vector<uint8_t>& k,const std::string& m){
    uint8_t r[32];unsigned len=32;HMAC(EVP_sha256(),k.data(),(int)k.size(),(const uint8_t*)m.data(),m.size(),r,&len);return{r,r+32};
}
static std::vector<uint8_t> hmacs(const std::string& k,const std::string& m){return hmacv({k.begin(),k.end()},m);}
static std::string utc_now(const char* fmt){time_t t=time(nullptr);struct tm buf;gmtime_r(&t,&buf);char b[32];strftime(b,sizeof(b),fmt,&buf);return b;}
static std::string uri_enc(const std::string& key){
    std::string r; for(unsigned char c:key){if(isalnum(c)||c=='-'||c=='_'||c=='.'||c=='~'||c=='/')r+=c;else{char buf[4];snprintf(buf,sizeof(buf),"%%%02X",c);r+=buf;}}return r;
}

struct S3Auth{std::string url,auth_hdr,date_hdr,sha_hdr;};

static void s3_build_url(const std::string& bucket,const std::string& key,const std::string& region, bool path_style, const char* ep_e, std::string& host,std::string& url,std::string& uri_path){
    if(ep_e){
        std::string ep=ep_e;
        if(ep.find("https://")==0)ep=ep.substr(8);
        if(ep.find("http://")==0) ep=ep.substr(7);
        if(path_style){host=ep;uri_path="/"+bucket+"/"+uri_enc(key);}
        else           {host=bucket+"."+ep;uri_path="/"+uri_enc(key);}
    } else {
        host=bucket+".s3."+region+".amazonaws.com"; uri_path="/"+uri_enc(key);
    }
    url="https://"+host+uri_path;
}

static S3Auth sign_s3_get(const std::string& bucket,const std::string& key,const std::string& region, const std::string& ak,const std::string& sk, uint64_t r0,uint64_t r1,const std::string& dt_ov=""){
    std::string dt=dt_ov.empty()?utc_now("%Y%m%dT%H%M%SZ"):dt_ov, date=dt.substr(0,8);
    const char* ep_e=getenv("AWS_S3_ENDPOINT"); const char* vh_e=getenv("AWS_VIRTUAL_HOSTING");
    bool ps=(vh_e&&std::string(vh_e)=="FALSE");
    std::string host,url,uri_path; s3_build_url(bucket,key,region,ps,ep_e,host,url,uri_path);
    std::string rng="bytes="+std::to_string(r0)+"-"+std::to_string(r1), ph=sha256h("");
    std::string ch="host:"+host+"\nrange:"+rng+"\nx-amz-content-sha256:"+ph+"\nx-amz-date:"+dt+"\n";
    std::string sh="host;range;x-amz-content-sha256;x-amz-date";
    std::string cr="GET\n"+uri_path+"\n\n"+ch+"\n"+sh+"\n"+ph;
    std::string scope=date+"/"+region+"/s3/aws4_request";
    std::string sts="AWS4-HMAC-SHA256\n"+dt+"\n"+scope+"\n"+sha256h(cr);
    auto kd=hmacs("AWS4"+sk,date),kr=hmacv(kd,region),ks=hmacv(kr,"s3"),kf=hmacv(ks,"aws4_request");
    std::string sig=hex_enc(hmacv(kf,sts).data(),32);
    return{url,"AWS4-HMAC-SHA256 Credential="+ak+"/"+scope+", SignedHeaders="+sh+", Signature="+sig,dt,ph};
}

static S3Auth sign_s3_put(const std::string& bucket,const std::string& key,const std::string& region, const std::string& ak,const std::string& sk, uint64_t content_len,const std::string& dt_ov=""){
    (void)content_len;
    std::string dt=dt_ov.empty()?utc_now("%Y%m%dT%H%M%SZ"):dt_ov, date=dt.substr(0,8);
    const char* ep_e=getenv("AWS_S3_ENDPOINT"); const char* vh_e=getenv("AWS_VIRTUAL_HOSTING");
    bool ps=(vh_e&&std::string(vh_e)=="FALSE");
    std::string host,url,uri_path; s3_build_url(bucket,key,region,ps,ep_e,host,url,uri_path);
    static const std::string up="UNSIGNED-PAYLOAD";
    std::string ch="content-type:application/octet-stream\nhost:"+host+"\nx-amz-content-sha256:"+up+"\nx-amz-date:"+dt+"\n";
    std::string sh="content-type;host;x-amz-content-sha256;x-amz-date";
    std::string cr="PUT\n"+uri_path+"\n\n"+ch+"\n"+sh+"\n"+up;
    std::string scope=date+"/"+region+"/s3/aws4_request";
    std::string sts="AWS4-HMAC-SHA256\n"+dt+"\n"+scope+"\n"+sha256h(cr);
    auto kd=hmacs("AWS4"+sk,date),kr=hmacv(kd,region),ks=hmacv(kr,"s3"),kf=hmacv(ks,"aws4_request");
    std::string sig=hex_enc(hmacv(kf,sts).data(),32);
    return{url,"AWS4-HMAC-SHA256 Credential="+ak+"/"+scope+", SignedHeaders="+sh+", Signature="+sig,dt,up};
}

static size_t curl_wcb(char* p,size_t sz,size_t nm,void* ud){ auto* v=(std::vector<uint8_t>*)ud; v->insert(v->end(),p,p+sz*nm); return sz*nm; }
struct CurlReadCtx{const uint8_t* data; size_t size; size_t pos;};
static size_t curl_rcb(void* buf,size_t sz,size_t nm,void* ud){
    auto* c=(CurlReadCtx*)ud; size_t want=sz*nm; size_t avail=c->size-c->pos; size_t copy=std::min(want,avail);
    memcpy(buf,c->data+c->pos,copy); c->pos+=copy; return copy;
}

static std::vector<uint8_t> s3_fetch_range(const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk,uint64_t r0,uint64_t r1){
    auto a=sign_s3_get(bucket,key,region,ak,sk,r0,r1);
    std::vector<uint8_t> buf; CURL* c=curl_easy_init(); if(!c)return buf;
    curl_slist* h=nullptr;
    h=curl_slist_append(h,("Authorization: "+a.auth_hdr).c_str());
    h=curl_slist_append(h,("x-amz-date: "+a.date_hdr).c_str());
    h=curl_slist_append(h,("x-amz-content-sha256: "+a.sha_hdr).c_str());
    h=curl_slist_append(h,("Range: bytes="+std::to_string(r0)+"-"+std::to_string(r1)).c_str());
    curl_easy_setopt(c,CURLOPT_URL,a.url.c_str());curl_easy_setopt(c,CURLOPT_HTTPHEADER,h);
    curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,curl_wcb);curl_easy_setopt(c,CURLOPT_WRITEDATA,&buf);
    curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,1L);curl_easy_perform(c);
    curl_slist_free_all(h);curl_easy_cleanup(c);return buf;
}

static void s3_put_file(const std::string& local_path,const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk){
    FILE* fp=fopen(local_path.c_str(),"rb"); if(!fp) throw std::runtime_error("Cannot open for S3 upload: "+local_path);
    fseek(fp,0,SEEK_END); long fsz=ftell(fp); rewind(fp);
    std::vector<uint8_t> fbuf((size_t)fsz); fread(fbuf.data(),1,(size_t)fsz,fp); fclose(fp);
    auto a=sign_s3_put(bucket,key,region,ak,sk,(uint64_t)fsz);
    std::vector<uint8_t> resp; CURL* c=curl_easy_init(); if(!c)throw std::runtime_error("curl_easy_init failed");
    CurlReadCtx rctx{fbuf.data(),(size_t)fsz,0}; curl_slist* h=nullptr;
    h=curl_slist_append(h,("Authorization: "+a.auth_hdr).c_str());
    h=curl_slist_append(h,("x-amz-date: "+a.date_hdr).c_str());
    h=curl_slist_append(h,("x-amz-content-sha256: "+a.sha_hdr).c_str());
    h=curl_slist_append(h,"Content-Type: application/octet-stream");
    h=curl_slist_append(h,("Content-Length: "+std::to_string(fsz)).c_str());
    curl_easy_setopt(c,CURLOPT_URL,a.url.c_str());curl_easy_setopt(c,CURLOPT_HTTPHEADER,h);
    curl_easy_setopt(c,CURLOPT_UPLOAD,1L);
    curl_easy_setopt(c,CURLOPT_READFUNCTION,curl_rcb);curl_easy_setopt(c,CURLOPT_READDATA,&rctx);
    curl_easy_setopt(c,CURLOPT_INFILESIZE_LARGE,(curl_off_t)fsz);
    curl_easy_setopt(c,CURLOPT_WRITEFUNCTION,curl_wcb);curl_easy_setopt(c,CURLOPT_WRITEDATA,&resp);
    curl_easy_setopt(c,CURLOPT_FOLLOWLOCATION,1L);
    CURLcode rc=curl_easy_perform(c); long http_code=0; curl_easy_getinfo(c,CURLINFO_RESPONSE_CODE,&http_code);
    curl_slist_free_all(h);curl_easy_cleanup(c);
    if(rc!=CURLE_OK||http_code/100!=2){ std::string body((char*)resp.data(),resp.size()); throw std::runtime_error("S3 PUT failed (HTTP "+std::to_string(http_code)+"): "+body.substr(0,256)); }
}

struct TileFetch{size_t idx;uint64_t offset,count;std::vector<uint8_t> data;};

static void s3_fetch_tiles(std::vector<TileFetch>& jobs,const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk, uint64_t gap=65536,int maxconn=64){
    if(jobs.empty())return;
    std::vector<size_t> order(jobs.size()); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return jobs[a].offset<jobs[b].offset;});
    struct MR{uint64_t start,end;std::vector<size_t> ji;std::vector<uint8_t> data;};
    std::vector<MR> ranges;
    for(size_t oi:order){
        uint64_t s=jobs[oi].offset,e=jobs[oi].offset+jobs[oi].count;
        if(!ranges.empty()&&s<=ranges.back().end+gap)ranges.back().end=std::max(ranges.back().end,e);
        else ranges.push_back({s,e,{},{}});
        ranges.back().ji.push_back(oi);
    }
    std::string dt=utc_now("%Y%m%dT%H%M%SZ"); CURLM* multi=curl_multi_init();
    curl_multi_setopt(multi,CURLMOPT_MAX_TOTAL_CONNECTIONS,(long)maxconn); curl_multi_setopt(multi,CURLMOPT_MAX_HOST_CONNECTIONS,(long)maxconn); curl_multi_setopt(multi,CURLMOPT_PIPELINING,CURLPIPE_MULTIPLEX);
    std::vector<CURL*> ev(ranges.size(),nullptr); std::vector<curl_slist*> hl(ranges.size(),nullptr);
    for(size_t i=0;i<ranges.size();i++){
        uint64_t r0=ranges[i].start,r1=ranges[i].end-1;
        auto a=sign_s3_get(bucket,key,region,ak,sk,r0,r1,dt);
        CURL* e=curl_easy_init(); if(!e)continue; ev[i]=e; auto& h=hl[i];
        h=curl_slist_append(h,("Authorization: "+a.auth_hdr).c_str());
        h=curl_slist_append(h,("x-amz-date: "+a.date_hdr).c_str());
        h=curl_slist_append(h,("x-amz-content-sha256: "+a.sha_hdr).c_str());
        h=curl_slist_append(h,("Range: bytes="+std::to_string(r0)+"-"+std::to_string(r1)).c_str());
        curl_easy_setopt(e,CURLOPT_URL,a.url.c_str());curl_easy_setopt(e,CURLOPT_HTTPHEADER,h);
        curl_easy_setopt(e,CURLOPT_WRITEFUNCTION,curl_wcb);curl_easy_setopt(e,CURLOPT_WRITEDATA,&ranges[i].data);
        curl_easy_setopt(e,CURLOPT_FOLLOWLOCATION,1L);curl_easy_setopt(e,CURLOPT_TCP_KEEPALIVE,1L);
        curl_easy_setopt(e,CURLOPT_HTTP_VERSION,CURL_HTTP_VERSION_2TLS);
        curl_multi_add_handle(multi,e);
    }
    int running=0;
    do{CURLMcode mc=curl_multi_perform(multi,&running); if(mc==CURLM_OK&&running)curl_multi_wait(multi,nullptr,0,100,nullptr); else if(mc!=CURLM_OK)break;}while(running);
    for(size_t i=0;i<ranges.size();i++){ if(ev[i]){curl_multi_remove_handle(multi,ev[i]);curl_easy_cleanup(ev[i]);} if(hl[i])curl_slist_free_all(hl[i]); }
    curl_multi_cleanup(multi);
    for(auto& rng:ranges){if(rng.data.empty())continue;
        for(size_t ji:rng.ji){auto& job=jobs[ji];uint64_t off=job.offset-rng.start;
            if(off+job.count>rng.data.size())continue;
            job.data.assign(rng.data.begin()+off,rng.data.begin()+off+job.count);}
    }
}

// ───────────────────────── TIFF PARSER ───────────────────────────────────────

static uint16_t ru16(const uint8_t* p,bool le){return le?(uint16_t)(p[0]|(p[1]<<8)):(uint16_t)((p[0]<<8)|p[1]);}
static uint32_t ru32(const uint8_t* p,bool le){
    if(le)return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); return((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
}
static uint64_t ru64(const uint8_t* p,bool le){uint64_t v=0; if(le){for(int i=0;i<8;i++)v|=((uint64_t)p[i]<<(8*i));} else  {for(int i=0;i<8;i++)v=(v<<8)|p[i];}return v;}
static size_t tiff_tsz(uint16_t t){ switch(t){case 1:case 2:case 6:case 7:return 1;case 3:case 8:return 2; case 4:case 9:case 11:return 4;case 5:case 10:case 12:case 16:case 17:case 18:return 8;default:return 1;} }
static uint64_t tiff_rval(const uint8_t* d,uint16_t type,bool le){ switch(type){case 1:case 6:return d[0];case 3:case 8:return ru16(d,le); case 4:case 9:return ru32(d,le);case 16:case 17:case 18:return ru64(d,le);default:return 0;} }
struct TiffTileIndex{std::vector<uint64_t> offsets,counts;uint16_t predictor=1,compression=1;};
struct TiffStripIndex{std::vector<uint64_t> offsets,counts;uint16_t predictor=1,compression=1;uint32_t rows_per_strip=0;int strips_per_band=0;};
struct TiffIFDView{ std::vector<uint8_t> head,ifd_extra; const uint8_t* hp=nullptr,*ifd_p=nullptr; uint64_t ifd_base=0;size_t ifd_sz=0; bool le=true,big=false;uint64_t nent=0,estart=0;size_t esz=12; };

static TiffIFDView load_tiff_ifd(const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk){
    TiffIFDView v; v.head=s3_fetch_range(bucket,key,region,ak,sk,0,131071);
    if(v.head.size()<8)throw std::runtime_error("S3 header fetch failed for s3://"+bucket+"/"+key);
    v.hp=v.head.data(); v.le=(v.hp[0]=='I'&&v.hp[1]=='I');
    if(!v.le&&!(v.hp[0]=='M'&&v.hp[1]=='M')){std::string s((char*)v.hp,std::min((size_t)256,v.head.size()));throw std::runtime_error("Not a valid TIFF: "+s);}
    uint16_t magic=ru16(v.hp+2,v.le); v.big=(magic==43);
    uint64_t ifd_off=v.big?ru64(v.hp+8,v.le):ru32(v.hp+4,v.le);
    v.ifd_p=v.hp;v.ifd_base=0;v.ifd_sz=v.head.size();
    if(ifd_off+(v.big?10u:4u)>v.head.size()){
        v.ifd_extra=s3_fetch_range(bucket,key,region,ak,sk,ifd_off,ifd_off+131071);
        if(v.ifd_extra.size()<(v.big?10u:4u))throw std::runtime_error("Failed to fetch IFD at "+std::to_string(ifd_off));
        v.ifd_p=v.ifd_extra.data();v.ifd_base=ifd_off;v.ifd_sz=v.ifd_extra.size();
    }
    uint64_t li=ifd_off-v.ifd_base; v.esz=v.big?20:12; v.nent=v.big?ru64(v.ifd_p+li,v.le):(uint64_t)ru16(v.ifd_p+li,v.le);
    v.estart=li+(v.big?8:2); return v;
}
static std::vector<uint64_t> tiff_read_array(const TiffIFDView& v,const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk,uint64_t doff,uint64_t cnt,uint16_t type){
    size_t ts=tiff_tsz(type),total=(size_t)cnt*ts; std::vector<uint64_t> vals(cnt);
    if(doff+total<=v.head.size()){for(uint64_t j=0;j<cnt;j++)vals[j]=tiff_rval(v.hp+doff+j*ts,type,v.le);}
    else if(!v.ifd_extra.empty()&&doff>=v.ifd_base&&doff+total<=v.ifd_base+v.ifd_sz){
        uint64_t lo=doff-v.ifd_base;for(uint64_t j=0;j<cnt;j++)vals[j]=tiff_rval(v.ifd_p+lo+j*ts,type,v.le);}
    else{auto arr=s3_fetch_range(bucket,key,region,ak,sk,doff,doff+total-1);
         if(arr.size()>=total)for(uint64_t j=0;j<cnt;j++)vals[j]=tiff_rval(arr.data()+j*ts,type,v.le);}
    return vals;
}
static TiffTileIndex parse_tiff_tile_index(const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk){
    auto v=load_tiff_ifd(bucket,key,region,ak,sk); TiffTileIndex idx;
    for(uint64_t i=0;i<v.nent;i++){
        uint64_t ep=v.estart+i*v.esz; if(ep+v.esz>v.ifd_sz)break;
        uint16_t tag=ru16(v.ifd_p+ep,v.le),type=ru16(v.ifd_p+ep+2,v.le);
        if(tag!=324&&tag!=325&&tag!=317&&tag!=259)continue;
        uint64_t cnt,vpos; if(!v.big){cnt=ru32(v.ifd_p+ep+4,v.le);vpos=ep+8;}else{cnt=ru64(v.ifd_p+ep+4,v.le);vpos=ep+12;}
        bool inl=v.big?(cnt*tiff_tsz(type)<=8):(cnt*tiff_tsz(type)<=4);
        if(tag==317){idx.predictor=(uint16_t)tiff_rval(v.ifd_p+vpos,type,v.le);continue;}
        if(tag==259){idx.compression=(uint16_t)tiff_rval(v.ifd_p+vpos,type,v.le);continue;}
        std::vector<uint64_t> vals;
        if(inl){vals.resize(cnt);vals[0]=tiff_rval(v.ifd_p+vpos,type,v.le);}
        else{uint64_t doff=v.big?ru64(v.ifd_p+vpos,v.le):ru32(v.ifd_p+vpos,v.le);vals=tiff_read_array(v,bucket,key,region,ak,sk,doff,cnt,type);}
        if(tag==324)idx.offsets=vals;else idx.counts=vals;
    }
    return idx;
}
static TiffStripIndex parse_tiff_strip_index(const std::string& bucket,const std::string& key, const std::string& region,const std::string& ak,const std::string& sk,int height){
    auto v=load_tiff_ifd(bucket,key,region,ak,sk); TiffStripIndex idx;
    for(uint64_t i=0;i<v.nent;i++){
        uint64_t ep=v.estart+i*v.esz; if(ep+v.esz>v.ifd_sz)break;
        uint16_t tag=ru16(v.ifd_p+ep,v.le),type=ru16(v.ifd_p+ep+2,v.le);
        if(tag!=273&&tag!=279&&tag!=278&&tag!=259&&tag!=317)continue;
        uint64_t cnt,vpos; if(!v.big){cnt=ru32(v.ifd_p+ep+4,v.le);vpos=ep+8;}else{cnt=ru64(v.ifd_p+ep+4,v.le);vpos=ep+12;}
        bool inl=v.big?(cnt*tiff_tsz(type)<=8):(cnt*tiff_tsz(type)<=4);
        if(tag==317){idx.predictor=(uint16_t)tiff_rval(v.ifd_p+vpos,type,v.le);continue;}
        if(tag==259){idx.compression=(uint16_t)tiff_rval(v.ifd_p+vpos,type,v.le);continue;}
        if(tag==278){idx.rows_per_strip=(uint32_t)tiff_rval(v.ifd_p+vpos,type,v.le);continue;}
        std::vector<uint64_t> vals;
        if(inl){vals.resize(cnt);vals[0]=tiff_rval(v.ifd_p+vpos,type,v.le);}
        else{uint64_t doff=v.big?ru64(v.ifd_p+vpos,v.le):ru32(v.ifd_p+vpos,v.le);vals=tiff_read_array(v,bucket,key,region,ak,sk,doff,cnt,type);}
        if(tag==273)idx.offsets=vals;else idx.counts=vals;
    }
    if(idx.rows_per_strip==0)idx.rows_per_strip=(uint32_t)height;
    idx.strips_per_band=((int)height+(int)idx.rows_per_strip-1)/(int)idx.rows_per_strip;
    return idx;
}

// ───────────────────────── DECOMP / UNPREDICT / CONVERT ──────────────────────

static std::vector<uint8_t> deflate_decomp(const uint8_t* src,size_t slen,size_t expected){
    std::vector<uint8_t> dst(expected); uLongf dlen=(uLongf)expected;
    if(uncompress(dst.data(),&dlen,src,(uLong)slen)==Z_OK){dst.resize(dlen);return dst;}
    z_stream zs{};zs.next_in=(Bytef*)src;zs.avail_in=(uInt)slen;zs.next_out=dst.data();zs.avail_out=(uInt)expected;
    if(inflateInit2(&zs,-15)==Z_OK){inflate(&zs,Z_FINISH);inflateEnd(&zs);dst.resize(zs.total_out);return dst;}
    return {};
}
static std::vector<uint8_t> packbits_decomp(const uint8_t* src,size_t slen,size_t expected){
    std::vector<uint8_t> dst;dst.reserve(expected);
    for(size_t i=0;i<slen&&dst.size()<expected;){
        int8_t n=(int8_t)src[i++];
        if(n>=0){size_t c=n+1;if(i+c>slen)break;dst.insert(dst.end(),src+i,src+i+c);i+=c;}
        else    {size_t c=-n+1;if(i>=slen)break;dst.insert(dst.end(),c,src[i++]);}
    }
    return dst;
}
static void unpredict2_u16(uint8_t* data,size_t rows,size_t vpr){
    uint16_t* p=(uint16_t*)data;
    for(size_t r=0;r<rows;r++){uint16_t* row=p+r*vpr;for(size_t c=1;c<vpr;c++)row[c]=(uint16_t)(row[c]+row[c-1]);}
}
static void unpredict3_f32(uint8_t* data,size_t rows,size_t vpr){
    std::vector<uint8_t> tmp(vpr*4);
    for(size_t r=0;r<rows;r++){
        uint8_t* row=data+r*vpr*4;
        for(int b=0;b<4;b++){uint8_t* pl=row+b*vpr;for(size_t i=1;i<vpr;i++)pl[i]=(uint8_t)(pl[i]+pl[i-1]);}
        for(size_t i=0;i<vpr;i++){tmp[i*4+0]=row[0*vpr+i];tmp[i*4+1]=row[1*vpr+i];tmp[i*4+2]=row[2*vpr+i];tmp[i*4+3]=row[3*vpr+i];}
        memcpy(row,tmp.data(),vpr*4);
    }
}

static void to_float_u16_avx2(const uint8_t* src,float* dst,size_t n){
#ifdef __AVX2__
    const uint16_t* s=(const uint16_t*)src; size_t i=0;
    for(;i+8<=n;i+=8){
        __m128i u16=_mm_loadu_si128((const __m128i*)(s+i));
        __m256i u32=_mm256_cvtepu16_epi32(u16);
        __m256  fp =_mm256_cvtepi32_ps(u32);
        _mm256_storeu_ps(dst+i,fp);
    }
    for(;i<n;i++) dst[i]=(float)s[i];
#else
    const uint16_t* s=(const uint16_t*)src;
    for(size_t i=0;i<n;i++) dst[i]=(float)s[i];
#endif
}

static void to_float(const uint8_t* src,float* dst,size_t n,GDALDataType dt){
    switch(dt){
        case GDT_Byte:    for(size_t i=0;i<n;i++) dst[i]=(float)src[i]; break;
        case GDT_UInt16:  to_float_u16_avx2(src,dst,n); break;
        case GDT_Int16:   for(size_t i=0;i<n;i++) dst[i]=(float)((const int16_t*)src)[i]; break;
        case GDT_UInt32:  for(size_t i=0;i<n;i++) dst[i]=(float)((const uint32_t*)src)[i]; break;
        case GDT_Float32: memcpy(dst,src,n*sizeof(float)); break;
        default:          for(size_t i=0;i<n;i++) dst[i]=0.f; break;
    }
}

static std::vector<uint8_t> decomp(const std::vector<uint8_t>& src_data,int comp,size_t exp){
    const uint8_t* src=src_data.data(); size_t slen=src_data.size();
    if(comp==1)return src_data;
    if(comp==8||comp==32946)return deflate_decomp(src,slen,exp);
    if(comp==32773)return packbits_decomp(src,slen,exp);
    if(comp==50000){
        size_t dsize=ZSTD_getFrameContentSize(src,slen);
        if(dsize==ZSTD_CONTENTSIZE_ERROR||dsize==ZSTD_CONTENTSIZE_UNKNOWN)dsize=exp;
        std::vector<uint8_t> dst(dsize); size_t r=ZSTD_decompress(dst.data(),dsize,src,slen);
        if(ZSTD_isError(r))return{}; dst.resize(r);return dst;
    }
    throw std::runtime_error("Unsupported compression: "+std::to_string(comp));
}

// ───────────────────────── TILE / STRIP EXTRACTORS ───────────────────────────

static void extract_tile_bands(const float* tile,int tr,int tc,int y0,int ch,int imw, int bx,int by,int spp,bool pxil, const std::vector<int>& slots,const std::vector<float*>& hb,int bp=-1){
    int tsr=tr*by,tsc=tc*bx,lr0=std::max(0,y0-tsr),lr1=std::min(by-1,y0+ch-1-tsr),lc1=std::min(bx-1,imw-tsc-1);
    for(int lr=lr0;lr<=lr1;lr++){int crow=tsr+lr-y0;
        for(int lc=0;lc<=lc1;lc++){int icol=tsc+lc;
            if(pxil){size_t pix=(size_t)lr*bx+lc;for(size_t s=0;s<slots.size();s++)hb[s][crow*imw+icol]=tile[pix*spp+slots[s]];}
            else{for(size_t s=0;s<slots.size();s++){if(slots[s]==bp){hb[s][crow*imw+icol]=tile[(size_t)lr*bx+lc];break;}}}
        }
    }
}

static void extract_strip_bands(const float* strip,int strip_first_row,int strip_rows, int y0,int cur_h,int imw,int spp,bool pxil, const std::vector<int>& slots,const std::vector<float*>& hb,int bp=-1){
    int or0=std::max(y0,strip_first_row),or1=std::min(y0+cur_h-1,strip_first_row+strip_rows-1);
    if(or0>or1)return;
    for(int row=or0;row<=or1;row++){
        int local_row=row-strip_first_row, chunk_row=row-y0;
        if(pxil){
            const float* src=strip+(size_t)local_row*imw*spp;
            for(size_t s=0;s<slots.size();s++){
                float* dst=hb[s]+(size_t)chunk_row*imw;
                int sl=slots[s];
#ifdef __AVX2__
                __m256i vstride=_mm256_mullo_epi32(_mm256_set_epi32(7,6,5,4,3,2,1,0),_mm256_set1_epi32(spp));
                __m256i vslot  =_mm256_set1_epi32(sl);
                __m256i vidx   =_mm256_add_epi32(vstride,vslot);
                int x=0;
                for(;x+8<=imw;x+=8){
                    __m256i idx=_mm256_add_epi32(vidx,_mm256_set1_epi32(x*spp));
                    __m256 v=_mm256_i32gather_ps(src,idx,4);
                    _mm256_storeu_ps(dst+x,v);
                }
                for(;x<imw;x++) dst[x]=src[(size_t)x*spp+sl];
#else
                for(int x=0;x<imw;x++) dst[x]=src[(size_t)x*spp+sl];
#endif
            }
        } else {
            for(size_t s=0;s<slots.size();s++){
                if(slots[s]==bp){
                    memcpy(hb[s]+(size_t)chunk_row*imw,strip+(size_t)local_row*imw,(size_t)imw*sizeof(float));break;
                }
            }
        }
    }
}

// ───────────────────────── PATH / METADATA ───────────────────────────────────

static bool is_s3_path(const std::string& p){ return (p.size()>=7&&p.substr(0,7)=="/vsis3/")||(p.size()>=5&&p.substr(0,5)=="s3://"); }
struct S3Loc{std::string bucket,key;};
static S3Loc parse_s3_loc(const std::string& path){
    S3Loc loc; std::string rest;
    if(path.size()>=7&&path.substr(0,7)=="/vsis3/")rest=path.substr(7);
    else if(path.size()>=5&&path.substr(0,5)=="s3://")rest=path.substr(5);
    else return loc;
    auto sl=rest.find('/');
    if(sl!=std::string::npos){loc.bucket=rest.substr(0,sl);loc.key=rest.substr(sl+1);}
    return loc;
}
struct FileInfo{double gt[6]={};int width=0,height=0;std::string proj,interleave,compression; int bx=0,by=0;GDALDataType dtype=GDT_Float32;int spp=1,predictor=1;};
static FileInfo get_metadata(const std::string& gdal_path){
    GDALAllRegister(); GDALDataset* ds=(GDALDataset*)GDALOpen(gdal_path.c_str(),GA_ReadOnly);
    if(!ds)throw std::runtime_error("GDALOpen failed: "+gdal_path);
    FileInfo f; f.width=ds->GetRasterXSize();f.height=ds->GetRasterYSize();f.spp=ds->GetRasterCount();
    ds->GetGeoTransform(f.gt); const char* pr=ds->GetProjectionRef();f.proj=pr?pr:"";
    GDALRasterBand* b1=ds->GetRasterBand(1);if(!b1)throw std::runtime_error("No RasterBand 1: "+gdal_path);
    b1->GetBlockSize(&f.bx,&f.by);f.dtype=b1->GetRasterDataType();
    const char* il=ds->GetMetadataItem("INTERLEAVE","IMAGE_STRUCTURE");f.interleave=il?il:"BAND";
    const char* co=ds->GetMetadataItem("COMPRESSION","IMAGE_STRUCTURE");f.compression=co?co:"NONE";
    const char* p2=ds->GetMetadataItem("PREDICTOR","IMAGE_STRUCTURE");f.predictor=p2?std::stoi(std::string(p2)):1;
    GDALClose(ds);return f;
}

// ───────────────────────── RASTER RESULT ─────────────────────────────────────

struct RasterResult {
    float* data      = nullptr;   
    bool        spilled   = false;
    std::string spill_path;
    int         width     = 0;
    int         height    = 0;
    double      gt[6]     = {};
    std::string proj;
    FileInfo    fi;

    size_t pixel_count()const{return (size_t)width*height;}
    size_t byte_count() const{return pixel_count()*sizeof(float);}

    void alloc(){
        if(data||spilled)return;
        size_t sz=byte_count();
        data=new(std::nothrow)float[pixel_count()];
        if(!data){
            char tmp[]="/dev/shm/curaster_XXXXXX.bin";
            int fd=mkstemps(tmp,4);
            if(fd<0){strcpy(tmp,"/tmp/curaster_XXXXXX.bin");fd=mkstemps(tmp,4);}
            if(fd<0)throw std::runtime_error("Cannot allocate RasterResult: no RAM and no tmpfs");
            ftruncate(fd,(off_t)sz);close(fd);
            data=(float*)malloc(sz);
            if(!data)throw std::runtime_error("Cannot allocate "+std::to_string(sz)+" bytes for result");
            spill_path=tmp; spilled=true;
        }
        memset(data,0,sz);
    }

    void flush_spill(){
        if(!spilled||!data||spill_path.empty())return;
        FILE* fp=fopen(spill_path.c_str(),"wb");
        if(fp){fwrite(data,sizeof(float),pixel_count(),fp);fclose(fp);}
    }
    void free_data(){if(data){delete[] data;data=nullptr;}}
    ~RasterResult(){free_data();}
    RasterResult()=default;
    RasterResult(const RasterResult&)=delete;
    RasterResult& operator=(const RasterResult&)=delete;
};

// ─────────────────────── SHARED ENGINE CORE ──────────────────────────────────

static void run_engine(
    const std::string& input_file,
    const std::vector<Instruction>& insts,
    const std::set<int>& bi,
    bool verbose,
    GDALRasterBand* out_band,   
    RasterResult* result)     
{
    S3Loc         s3_loc;
    TiffTileIndex s3_tile_idx;
    TiffStripIndex s3_strip_idx;
    bool           s3_is_tiled=true;
    std::string    s3_ak,s3_sk,s3_region;
    bool using_s3=is_s3_path(input_file);

    if(using_s3){
        s3_loc=parse_s3_loc(input_file);
        const char* ak_e=getenv("AWS_ACCESS_KEY_ID"),*sk_e=getenv("AWS_SECRET_ACCESS_KEY"),*rg_e=getenv("AWS_REGION");
        if(!ak_e||!sk_e||!rg_e)throw std::runtime_error("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION must be set.");
        s3_ak=ak_e;s3_sk=sk_e;s3_region=rg_e;
    }

    std::vector<int> slots(bi.begin(),bi.end()); int nb=(int)slots.size();
    FileInfo fi=get_metadata(input_file);

    if(using_s3){
        s3_tile_idx=parse_tiff_tile_index(s3_loc.bucket,s3_loc.key,s3_region,s3_ak,s3_sk);
        if(s3_tile_idx.offsets.empty()){
            s3_is_tiled=false;
            s3_strip_idx=parse_tiff_strip_index(s3_loc.bucket,s3_loc.key,s3_region,s3_ak,s3_sk,fi.height);
            if(s3_strip_idx.offsets.empty())throw std::runtime_error("No tile/strip offsets — corrupt TIFF?");
            if(fi.predictor!=1)s3_strip_idx.predictor=(uint16_t)fi.predictor;
        } else {if(fi.predictor!=1)s3_tile_idx.predictor=(uint16_t)fi.predictor;}
    }

    size_t ta=((size_t)fi.width+fi.bx-1)/fi.bx, td=((size_t)fi.height+fi.by-1)/fi.by;
    bool pxil=(fi.interleave=="PIXEL");

    size_t avail=get_avail_ram();
    size_t gdal_cache=(size_t)(avail*0.10), pinned_bgt=(size_t)(avail*0.40);
    CPLSetConfigOption("GDAL_CACHEMAX",std::to_string(gdal_cache/(1024*1024)).c_str());

    size_t bpp=((size_t)nb+1)*sizeof(float), row_total=(size_t)fi.width*bpp*kNumThreads;
    int max_rows=(int)(pinned_bgt/row_total);

    int ckhgt;
    if(using_s3&&s3_is_tiled){
        int tdc=(int)td,tc2=std::max(kNumThreads*2,tdc),trpc=std::max(1,tdc/tc2);
        ckhgt=trpc*fi.by;
        if(max_rows>0&&ckhgt>max_rows)ckhgt=std::max(fi.by,(max_rows/fi.by)*fi.by);
        if(ckhgt<fi.by)ckhgt=fi.by;
    } else if(using_s3&&!s3_is_tiled){
        int rps=(int)s3_strip_idx.rows_per_strip;
        ckhgt=rps; while(ckhgt+rps<=max_rows)ckhgt+=rps; if(ckhgt<rps)ckhgt=rps;
    } else {
        ckhgt=fi.by; while(ckhgt+fi.by<=max_rows)ckhgt+=fi.by; if(ckhgt<fi.by)ckhgt=fi.by;
    }

    if(verbose){
        if(using_s3&&s3_is_tiled)
            printf("[S3 Tiled] %s/%s  %dx%d  tiles %zux%zu  block %dx%d  chunk_h=%d  pred=%d  comp=%d  il=%s  dt=%s  bands=%d\n",
                   s3_loc.bucket.c_str(),s3_loc.key.c_str(),fi.width,fi.height,ta,td,fi.bx,fi.by,ckhgt,
                   s3_tile_idx.predictor,s3_tile_idx.compression,fi.interleave.c_str(),GDALGetDataTypeName(fi.dtype),nb);
        else if(using_s3&&!s3_is_tiled)
            printf("[S3 Strip] %s/%s  %dx%d  rps=%d  spb=%d  chunk_h=%d  pred=%d  comp=%d  il=%s  dt=%s  bands=%d\n",
                   s3_loc.bucket.c_str(),s3_loc.key.c_str(),fi.width,fi.height,
                   s3_strip_idx.rows_per_strip,s3_strip_idx.strips_per_band,ckhgt,
                   s3_strip_idx.predictor,s3_strip_idx.compression,fi.interleave.c_str(),GDALGetDataTypeName(fi.dtype),nb);
        else
            printf("[Local] %dx%d  block %dx%d  chunk_h=%d  interleave=%s\n",
                   fi.width,fi.height,fi.bx,fi.by,ckhgt,fi.interleave.c_str());
        fflush(stdout);
    }

    size_t max_pix=(size_t)fi.width*ckhgt, band_bytes=max_pix*sizeof(float);
    std::vector<ThreadBufs> pool(kNumThreads);
    for(int t=0;t<kNumThreads;t++) pool[t].alloc(band_bytes,nb);
    int num_chunks=(fi.height+ckhgt-1)/ckhgt;

    Instruction* d_prog;
    cudaCheck(cudaMalloc(&d_prog,insts.size()*sizeof(Instruction)));
    cudaCheck(cudaMemcpy(d_prog,insts.data(),insts.size()*sizeof(Instruction),cudaMemcpyHostToDevice));

    GDALDatasetContainer container(kNumThreads,nb);
    std::vector<int> band_map;
    if(!using_s3){
        for(int i=0;i<kNumThreads;++i){
            container.in_datasets[i]=(GDALDataset*)GDALOpen(input_file.c_str(),GA_ReadOnly);
            int idx=0;
            for(int band:bi){container.bands[i][idx]=container.in_datasets[i]->GetRasterBand(band+1);if(i==0)band_map.push_back(band+1);idx++;}
        }
    }

    omp_set_num_threads(kNumThreads);
    std::atomic<int> done{0}; double ps=-1.0;
    if(verbose)GDALTermProgress(0.0,nullptr,&ps);

    #pragma omp parallel for schedule(dynamic,1)
    for(int chunk=0;chunk<num_chunks;chunk++){
        int tid=omp_get_thread_num(), y0=chunk*ckhgt, cur_h=std::min(ckhgt,fi.height-y0);
        size_t pixels=(size_t)fi.width*cur_h;
        ThreadBufs& buf=pool[tid];
        memset(buf.h_master,0,band_bytes*nb);

        if(!using_s3){
            if(chunk+2<num_chunks){int ay=(chunk+2)*ckhgt,ah=std::min(ckhgt,fi.height-ay);
                (void)container.in_datasets[tid]->AdviseRead(0,ay,fi.width,ah,fi.width,ah,GDT_Float32,0,nullptr,nullptr);}
            if(pxil) (void)container.in_datasets[tid]->RasterIO(GF_Read,0,y0,fi.width,cur_h,buf.h_master,fi.width,cur_h,GDT_Float32,(int)band_map.size(),band_map.data(),sizeof(float),(size_t)fi.width*sizeof(float),band_bytes,nullptr);
            else for(int b=0;b<nb;b++) (void)container.bands[tid][b]->RasterIO(GF_Read,0,y0,fi.width,cur_h,buf.h_bands[b],fi.width,cur_h,GDT_Float32,0,0);
        } else if(s3_is_tiled){
            int tr0=y0/fi.by,tr1=(y0+cur_h-1)/fi.by;
            int comp=s3_tile_idx.compression,pred=s3_tile_idx.predictor;
            size_t bps=(size_t)GDALGetDataTypeSizeBytes(fi.dtype);
            std::vector<TileFetch> jobs;
            if(pxil){for(int tr=tr0;tr<=tr1;tr++)for(size_t tc=0;tc<ta;tc++){size_t ti=(size_t)tr*ta+tc;if(ti<s3_tile_idx.offsets.size()&&ti<s3_tile_idx.counts.size()&&s3_tile_idx.counts[ti]>0)jobs.push_back({ti,s3_tile_idx.offsets[ti],s3_tile_idx.counts[ti],{}});}}
            else{for(int s=0;s<nb;s++){int b=slots[s];for(int tr=tr0;tr<=tr1;tr++)for(size_t tc=0;tc<ta;tc++){size_t ti=(size_t)b*ta*td+(size_t)tr*ta+tc;if(ti<s3_tile_idx.offsets.size()&&ti<s3_tile_idx.counts.size()&&s3_tile_idx.counts[ti]>0)jobs.push_back({ti,s3_tile_idx.offsets[ti],s3_tile_idx.counts[ti],{}});}}}
            s3_fetch_tiles(jobs,s3_loc.bucket,s3_loc.key,s3_region,s3_ak,s3_sk);
            for(auto& job:jobs){if(job.data.empty())continue;
                int tr_j,tc_j,bp_j;
                if(pxil){tr_j=(int)(job.idx/ta);tc_j=(int)(job.idx%ta);bp_j=-1;}
                else{size_t pt=ta*td;bp_j=(int)(job.idx/pt);tr_j=(int)((job.idx%pt)/ta);tc_j=(int)((job.idx%pt)%ta);}
                size_t vpr=pxil?(size_t)fi.bx*fi.spp:(size_t)fi.bx;
                size_t exp=(size_t)fi.bx*fi.by*(pxil?fi.spp:1)*bps;
                auto raw=decomp(job.data,comp,exp); if(raw.empty())continue;
                if(raw.size()<exp)raw.resize(exp,0);
                if(pred==2)unpredict2_u16(raw.data(),fi.by,vpr);
                else if(pred==3)unpredict3_f32(raw.data(),fi.by,vpr);
                size_t nv=(size_t)fi.bx*fi.by*(pxil?fi.spp:1); std::vector<float> tf(nv);
                to_float(raw.data(),tf.data(),nv,fi.dtype);
                extract_tile_bands(tf.data(),tr_j,tc_j,y0,cur_h,fi.width,fi.bx,fi.by,fi.spp,pxil,slots,buf.h_bands,bp_j);
            }
        } else {
            int rps=(int)s3_strip_idx.rows_per_strip,sr0=y0/rps,sr1=(y0+cur_h-1)/rps;
            int comp=s3_strip_idx.compression,pred=s3_strip_idx.predictor;
            size_t bps=(size_t)GDALGetDataTypeSizeBytes(fi.dtype);
            std::vector<TileFetch> jobs;
            if(pxil){for(int sr=sr0;sr<=sr1;sr++){size_t si=(size_t)sr;if(si<s3_strip_idx.offsets.size()&&si<s3_strip_idx.counts.size()&&s3_strip_idx.counts[si]>0)jobs.push_back({si,s3_strip_idx.offsets[si],s3_strip_idx.counts[si],{}});}}
            else{for(int s=0;s<nb;s++){int b=slots[s];for(int sr=sr0;sr<=sr1;sr++){size_t si=(size_t)b*s3_strip_idx.strips_per_band+sr;if(si<s3_strip_idx.offsets.size()&&si<s3_strip_idx.counts.size()&&s3_strip_idx.counts[si]>0)jobs.push_back({si,s3_strip_idx.offsets[si],s3_strip_idx.counts[si],{}});}}}
            s3_fetch_tiles(jobs,s3_loc.bucket,s3_loc.key,s3_region,s3_ak,s3_sk);
            for(auto& job:jobs){if(job.data.empty())continue;
                int sr_j,bp_j;
                if(pxil){sr_j=(int)job.idx;bp_j=-1;}
                else{sr_j=(int)(job.idx%(size_t)s3_strip_idx.strips_per_band);bp_j=(int)(job.idx/(size_t)s3_strip_idx.strips_per_band);}
                int sfr=sr_j*rps, srows=std::min(rps,fi.height-sfr);
                size_t vpr=pxil?(size_t)fi.width*fi.spp:(size_t)fi.width;
                size_t exp=(size_t)fi.width*srows*(pxil?fi.spp:1)*bps;
                auto raw=decomp(job.data,comp,exp); if(raw.empty())continue;
                if(raw.size()<exp)raw.resize(exp,0);
                if(pred==2)unpredict2_u16(raw.data(),(size_t)srows,vpr);
                else if(pred==3)unpredict3_f32(raw.data(),(size_t)srows,vpr);
                size_t nv=(size_t)fi.width*srows*(pxil?fi.spp:1); std::vector<float> tf(nv);
                to_float(raw.data(),tf.data(),nv,fi.dtype);
                extract_strip_bands(tf.data(),sfr,srows,y0,cur_h,fi.width,fi.spp,pxil,slots,buf.h_bands,bp_j);
            }
        }

        // --- C++ LAUNCHES THE GPU KERNEL VIA THE BRIDGE HEADER ---
        launch_raster_algebra(d_prog, (int)insts.size(), buf.d_band_ptrs, buf.d_out, pixels, buf.stream);
        cudaCheck(cudaStreamSynchronize(buf.stream));

        #pragma omp critical
        {
            if(out_band)
                (void)out_band->RasterIO(GF_Write,0,y0,fi.width,cur_h,buf.h_out,fi.width,cur_h,GDT_Float32,0,0);
            else if(result&&result->data)
                memcpy(result->data+(size_t)y0*fi.width, buf.h_out, pixels*sizeof(float));
            int nd=++done;
            if(verbose){GDALTermProgress((double)nd/num_chunks,nullptr,&ps);fflush(stdout);}
        }
    }

    for(int t=0;t<kNumThreads;t++){if(!using_s3)GDALClose(container.in_datasets[t]);pool[t].free_all();}
    cudaFree(d_prog);
}

// ─────────────── GDAL OUTPUT DATASET HELPER ──────────────────────────────────

static GDALDataset* create_output_ds(const std::string& path,const FileInfo& fi){
    GDALDriver* drv=GetGDALDriverManager()->GetDriverByName("GTiff");
    char** opts=nullptr;
    opts=CSLSetNameValue(opts,"COMPRESS","ZSTD"); opts=CSLSetNameValue(opts,"ZSTD_LEVEL","1");
    opts=CSLSetNameValue(opts,"NUM_THREADS","ALL_CPUS"); opts=CSLSetNameValue(opts,"TILED","YES");
    opts=CSLSetNameValue(opts,"BLOCKXSIZE","512"); opts=CSLSetNameValue(opts,"BLOCKYSIZE","512");
    opts=CSLSetNameValue(opts,"INTERLEAVE","BAND"); opts=CSLSetNameValue(opts,"BIGTIFF","IF_SAFER");
    GDALDataset* ds=drv->Create(path.c_str(),fi.width,fi.height,1,GDT_Float32,opts);
    ds->SetGeoTransform(const_cast<double*>(fi.gt)); ds->SetProjection(fi.proj.c_str());
    ds->GetRasterBand(1)->SetNoDataValue(-9999.0); CSLDestroy(opts); return ds;
}

// ───────────────────────── PUBLIC API ────────────────────────────────────────

void compute_algebra(const std::string& input_file,const std::string& output_file,
                     const std::string& expression,bool verbose){
    std::set<int> bi;
    auto insts=compile_rpn(tokenize_rpn(expression,bi),bi);
    FileInfo fi=get_metadata(input_file);
    GDALDataset* ods=create_output_ds(output_file,fi);
    GDALRasterBand* ob=ods->GetRasterBand(1);
    run_engine(input_file,insts,bi,verbose,ob,nullptr);
    GDALClose(ods);
}

std::shared_ptr<RasterResult> reclassify(const std::string& input_file,
                                          const std::string& expression,
                                          bool verbose){
    std::set<int> bi;
    auto insts=compile_reclass(tokenize_reclass(expression,bi),bi);
    FileInfo fi=get_metadata(input_file);
    auto res=std::make_shared<RasterResult>();
    res->width=fi.width; res->height=fi.height; res->fi=fi; res->proj=fi.proj;
    memcpy(res->gt,fi.gt,sizeof(fi.gt));
    res->alloc();
    run_engine(input_file,insts,bi,verbose,nullptr,res.get());
    return res;
}

void write_local(std::shared_ptr<RasterResult> res,const std::string& output_path){
    if(!res||!res->data) throw std::runtime_error("write_local: null or empty RasterResult");
    GDALDataset* ods=create_output_ds(output_path,res->fi);
    GDALRasterBand* ob=ods->GetRasterBand(1);
    (void)ob->RasterIO(GF_Write,0,0,res->width,res->height,res->data,res->width,res->height,GDT_Float32,0,0);
    GDALClose(ods);
}

void write_s3_output(std::shared_ptr<RasterResult> res,const std::string& s3_path){
    if(!res||!res->data) throw std::runtime_error("write_s3_output: null or empty RasterResult");
    const char* ak_e=getenv("AWS_ACCESS_KEY_ID"),*sk_e=getenv("AWS_SECRET_ACCESS_KEY"),*rg_e=getenv("AWS_REGION");
    if(!ak_e||!sk_e||!rg_e)throw std::runtime_error("AWS env vars not set");
    std::string ak=ak_e,sk=sk_e,region=rg_e;
    auto loc=parse_s3_loc(s3_path);
    if(loc.bucket.empty()||loc.key.empty())throw std::runtime_error("Invalid S3 path: "+s3_path);

    char tmp[]="/dev/shm/curaster_out_XXXXXX.tif"; int fd=mkstemps(tmp,4);
    if(fd<0){strcpy(tmp,"/tmp/curaster_out_XXXXXX.tif");fd=mkstemps(tmp,4);}
    if(fd<0)throw std::runtime_error("Cannot create temp file for S3 upload");
    close(fd);
    std::string tmp_path=tmp;

    write_local(res,tmp_path);
    try{
        s3_put_file(tmp_path,loc.bucket,loc.key,region,ak,sk);
    } catch(...){ unlink(tmp_path.c_str()); throw; }
    unlink(tmp_path.c_str());
}

// ───────────────────────── PYTHON BINDINGS ───────────────────────────────────

PYBIND11_MODULE(curaster,m){
    pybind11::module_ os=pybind11::module_::import("os");
    std::string mdir=os.attr("path").attr("dirname")(m.attr("__file__")).cast<std::string>();
    std::string proj_path=mdir+"/proj_data", gdal_path_d=mdir+"/gdal_data";
    if(os.attr("path").attr("exists")(proj_path).cast<bool>()){
        os.attr("environ")["PROJ_LIB"]=proj_path; os.attr("environ")["PROJ_DATA"]=proj_path;
        CPLSetConfigOption("PROJ_LIB",proj_path.c_str()); CPLSetConfigOption("PROJ_DATA",proj_path.c_str());
    }
    if(os.attr("path").attr("exists")(gdal_path_d).cast<bool>()){
        os.attr("environ")["GDAL_DATA"]=gdal_path_d; CPLSetConfigOption("GDAL_DATA",gdal_path_d.c_str());
    }
    curl_global_init(CURL_GLOBAL_ALL);

    pybind11::class_<RasterResult,std::shared_ptr<RasterResult>>(m,"RasterResult")
        .def_readonly("width",  &RasterResult::width)
        .def_readonly("height", &RasterResult::height)
        .def("save_local",[](std::shared_ptr<RasterResult> r,const std::string& p){write_local(r,p);})
        .def("save_s3",   [](std::shared_ptr<RasterResult> r,const std::string& p){write_s3_output(r,p);});

    m.def("compute",&compute_algebra,
        pybind11::arg("input_file"),pybind11::arg("output_file"),
        pybind11::arg("expression"),pybind11::arg("verbose")=true);

    m.def("reclassify",&reclassify,
        pybind11::arg("input_file"),pybind11::arg("expression"),
        pybind11::arg("verbose")=true);

    m.def("write_local",&write_local,
        pybind11::arg("result"),pybind11::arg("output_path"));
    m.def("write_s3",&write_s3_output,
        pybind11::arg("result"),pybind11::arg("s3_path"));
}
