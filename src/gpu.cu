#include "raster_core.h"
#include <cuda_runtime.h>
#include <math_functions.h>

__device__ inline float4 f4add(float4 a,float4 b){return make_float4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w);}
__device__ inline float4 f4sub(float4 a,float4 b){return make_float4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w);}
__device__ inline float4 f4mul(float4 a,float4 b){return make_float4(a.x*b.x,a.y*b.y,a.z*b.z,a.w*b.w);}
__device__ inline float4 f4div(float4 a,float4 b){
    return make_float4(__fdividef(a.x,b.x+1e-6f),__fdividef(a.y,b.y+1e-6f),
                       __fdividef(a.z,b.z+1e-6f),__fdividef(a.w,b.w+1e-6f));
}
__device__ inline float4 f4gt (float4 a,float4 b){return make_float4(a.x>b.x?1.f:0.f,a.y>b.y?1.f:0.f,a.z>b.z?1.f:0.f,a.w>b.w?1.f:0.f);}
__device__ inline float4 f4lt (float4 a,float4 b){return make_float4(a.x<b.x?1.f:0.f,a.y<b.y?1.f:0.f,a.z<b.z?1.f:0.f,a.w<b.w?1.f:0.f);}
__device__ inline float4 f4gte(float4 a,float4 b){return make_float4(a.x>=b.x?1.f:0.f,a.y>=b.y?1.f:0.f,a.z>=b.z?1.f:0.f,a.w>=b.w?1.f:0.f);}
__device__ inline float4 f4lte(float4 a,float4 b){return make_float4(a.x<=b.x?1.f:0.f,a.y<=b.y?1.f:0.f,a.z<=b.z?1.f:0.f,a.w<=b.w?1.f:0.f);}
__device__ inline float4 f4eq (float4 a,float4 b){return make_float4(a.x==b.x?1.f:0.f,a.y==b.y?1.f:0.f,a.z==b.z?1.f:0.f,a.w==b.w?1.f:0.f);}
__device__ inline float4 f4neq(float4 a,float4 b){return make_float4(a.x!=b.x?1.f:0.f,a.y!=b.y?1.f:0.f,a.z!=b.z?1.f:0.f,a.w!=b.w?1.f:0.f);}
__device__ inline float4 f4and(float4 a,float4 b){return f4mul(a,b);}
__device__ inline float4 f4or (float4 a,float4 b){return make_float4(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w));}
__device__ inline float4 f4not(float4 a)          {return f4sub(make_float4(1,1,1,1),a);}
__device__ inline float4 f4if(float4 cond,float4 tv,float4 fv){
    return f4add(f4mul(cond,tv),f4mul(f4not(cond),fv));
}
__device__ inline float4 f4min(float4 a,float4 b){return make_float4(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z),fminf(a.w,b.w));}
__device__ inline float4 f4max(float4 a,float4 b){return make_float4(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z),fmaxf(a.w,b.w));}

__global__ void kernel_raster_algebra(
    const Instruction* __restrict__ prog, int np,
    const float* const* __restrict__ bands,
    float* __restrict__ out, size_t sz)
{
    size_t base   = ((size_t)blockIdx.x*blockDim.x+threadIdx.x)*4;
    size_t stride = (size_t)gridDim.x*blockDim.x*4;
    while (base+3 < sz) {
        float4 stk[24]; int sp=0;
        for(int i=0;i<np;i++){
            const Instruction& ins=prog[i];
            if      (ins.op==OP_LOAD_CONST) stk[sp++]=make_float4(ins.constant,ins.constant,ins.constant,ins.constant);
            else if (ins.op==OP_LOAD_BAND)  stk[sp++]=__ldg((const float4*)(bands[ins.band_index]+base));
            else if (ins.op==OP_ADD) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4add(a,b);}
            else if (ins.op==OP_SUB) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4sub(a,b);}
            else if (ins.op==OP_MUL) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4mul(a,b);}
            else if (ins.op==OP_DIV) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4div(a,b);}
            else if (ins.op==OP_GT)  {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4gt(a,b);}
            else if (ins.op==OP_LT)  {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4lt(a,b);}
            else if (ins.op==OP_GTE) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4gte(a,b);}
            else if (ins.op==OP_LTE) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4lte(a,b);}
            else if (ins.op==OP_EQ)  {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4eq(a,b);}
            else if (ins.op==OP_NEQ) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4neq(a,b);}
            else if (ins.op==OP_AND) {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4and(a,b);}
            else if (ins.op==OP_OR)  {float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4or(a,b);}
            else if (ins.op==OP_NOT) {stk[sp-1]=f4not(stk[sp-1]);}
            else if (ins.op==OP_IF)  {float4 fv=stk[--sp],tv=stk[--sp],cond=stk[--sp];stk[sp++]=f4if(cond,tv,fv);}
            else if (ins.op==OP_BETWEEN){
                float4 hi=stk[--sp],lo=stk[--sp],x=stk[--sp];
                stk[sp++]=f4and(f4gte(x,lo),f4lte(x,hi));
            }
            else if (ins.op==OP_CLAMP){
                float4 hi=stk[--sp],lo=stk[--sp],x=stk[--sp];
                stk[sp++]=f4max(lo,f4min(x,hi));
            }
            else if (ins.op==OP_MIN2){float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4min(a,b);}
            else if (ins.op==OP_MAX2){float4 b=stk[--sp],a=stk[--sp];stk[sp++]=f4max(a,b);}
        }
        *((float4*)(out+base)) = sp>0?stk[0]:make_float4(0,0,0,0);
        base+=stride;
    }
    size_t tail=(sz/4)*4+(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(tail<sz){
        float stk[24]; int sp=0;
        for(int i=0;i<np;i++){
            const Instruction& ins=prog[i];
            if      (ins.op==OP_LOAD_CONST) stk[sp++]=ins.constant;
            else if (ins.op==OP_LOAD_BAND)  stk[sp++]=__ldg(&bands[ins.band_index][tail]);
            else if (ins.op==OP_ADD) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l+r;}
            else if (ins.op==OP_SUB) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l-r;}
            else if (ins.op==OP_MUL) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l*r;}
            else if (ins.op==OP_DIV) {float r=stk[--sp],l=stk[--sp];stk[sp++]=__fdividef(l,r+1e-6f);}
            else if (ins.op==OP_GT)  {float r=stk[--sp],l=stk[--sp];stk[sp++]=l>r?1.f:0.f;}
            else if (ins.op==OP_LT)  {float r=stk[--sp],l=stk[--sp];stk[sp++]=l<r?1.f:0.f;}
            else if (ins.op==OP_GTE) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l>=r?1.f:0.f;}
            else if (ins.op==OP_LTE) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l<=r?1.f:0.f;}
            else if (ins.op==OP_EQ)  {float r=stk[--sp],l=stk[--sp];stk[sp++]=l==r?1.f:0.f;}
            else if (ins.op==OP_NEQ) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l!=r?1.f:0.f;}
            else if (ins.op==OP_AND) {float r=stk[--sp],l=stk[--sp];stk[sp++]=l*r;}
            else if (ins.op==OP_OR)  {float r=stk[--sp],l=stk[--sp];stk[sp++]=fmaxf(l,r);}
            else if (ins.op==OP_NOT) {stk[sp-1]=1.f-stk[sp-1];}
            else if (ins.op==OP_IF)  {float fv=stk[--sp],tv=stk[--sp],cond=stk[--sp];stk[sp++]=cond*tv+(1.f-cond)*fv;}
            else if (ins.op==OP_BETWEEN){float hi=stk[--sp],lo=stk[--sp],x=stk[--sp];stk[sp++]=(x>=lo&&x<=hi)?1.f:0.f;}
            else if (ins.op==OP_CLAMP)  {float hi=stk[--sp],lo=stk[--sp],x=stk[--sp];stk[sp++]=fmaxf(lo,fminf(x,hi));}
            else if (ins.op==OP_MIN2){float r=stk[--sp],l=stk[--sp];stk[sp++]=fminf(l,r);}
            else if (ins.op==OP_MAX2){float r=stk[--sp],l=stk[--sp];stk[sp++]=fmaxf(l,r);}
        }
        out[tail]=sp>0?stk[0]:0.f;
    }
}

void launch_raster_algebra(
    const Instruction* d_prog, int np,
    const float* const* d_bands,
    float* d_out, size_t sz,
    cudaStream_t stream) 
{
    const int kTPB = 256;
    size_t groups = (sz + 3) / 4; 
    int blocks = (int)((groups + kTPB - 1) / kTPB);
    kernel_raster_algebra<<<blocks, kTPB, 0, stream>>>(d_prog, np, d_bands, d_out, sz);
}