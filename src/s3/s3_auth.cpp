/**
 * @file s3_auth.cpp
 * @brief AWS SigV4 HMAC-SHA256 request signing and S3 URI parsing.
 */
#include "s3_auth.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include <openssl/hmac.h>
#include <openssl/sha.h>



/// Encode @p num_bytes raw bytes as a lower-case hex string.
static std::string bytes_to_hex(const unsigned char* bytes, int num_bytes) {
    static const char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(num_bytes * 2);
    for (int i = 0; i < num_bytes; ++i) {
        result += hex_chars[bytes[i] >> 4];
        result += hex_chars[bytes[i] & 0x0F];
    }
    return result;
}

/// Return the lower-case hex-encoded SHA-256 digest of @p input.
static std::string sha256_hex(const std::string& input) {
    unsigned char digest[32];
    SHA256(reinterpret_cast<const unsigned char*>(input.data()),
           input.size(), digest);
    return bytes_to_hex(digest, 32);
}

/// Return the raw (binary) HMAC-SHA256 of @p message keyed with @p key.
static std::string hmac_sha256_raw(const std::string& key, const std::string& message) {
    unsigned char digest[32];
    unsigned int  digest_len = 32;
    HMAC(EVP_sha256(),
         key.data(),   static_cast<int>(key.size()),
         reinterpret_cast<const unsigned char*>(message.data()), message.size(),
         digest, &digest_len);
    return std::string(reinterpret_cast<char*>(digest), 32);
}



bool is_s3_path(const std::string& path) {
    return path.rfind("s3://",   0) == 0
        || path.rfind("/vsis3/", 0) == 0;
}

S3Loc parse_s3_path(const std::string& path) {
    
    std::string stripped = path;
    if (stripped.rfind("/vsis3/", 0) == 0) {
        stripped = stripped.substr(7);
    } else if (stripped.rfind("s3://", 0) == 0) {
        stripped = stripped.substr(5);
    }

    
    auto slash_pos    = stripped.find('/');
    S3Loc location;
    location.bucket   = stripped.substr(0, slash_pos);
    location.key      = stripped.substr(slash_pos + 1);

    
    const char* env_access_key    = std::getenv("AWS_ACCESS_KEY_ID");
    const char* env_secret_key    = std::getenv("AWS_SECRET_ACCESS_KEY");
    const char* env_session_token = std::getenv("AWS_SESSION_TOKEN");
    const char* env_region        = std::getenv("AWS_DEFAULT_REGION");
    const char* env_endpoint      = std::getenv("AWS_ENDPOINT_URL");

    location.access_key_id  = env_access_key    ? env_access_key    : "";
    location.secret_key     = env_secret_key     ? env_secret_key    : "";
    location.session_token  = env_session_token  ? env_session_token : "";
    location.region         = env_region         ? env_region        : "us-east-1";
    location.endpoint       = env_endpoint
                                  ? env_endpoint
                                  : ("s3." + location.region + ".amazonaws.com");
    location.is_anonymous   = location.access_key_id.empty();
    return location;
}



std::string build_s3_request_url(const S3Loc&       location,
                                 const std::string& range_header,
                                 std::string&       out_auth_header) {
    
    if (location.is_anonymous) {
        out_auth_header = range_header;
        return "https://" + location.endpoint + "/" + location.bucket + "/" + location.key;
    }

    
    auto now         = std::chrono::system_clock::now();
    auto time_t_now  = std::chrono::system_clock::to_time_t(now);
    struct tm utc_time{};
#ifdef _WIN32
    gmtime_s(&utc_time, &time_t_now);
#else
    gmtime_r(&time_t_now, &utc_time);
#endif

    char date8[9];   
    char dt16[17];   
    strftime(date8, sizeof(date8), "%Y%m%d",        &utc_time);
    strftime(dt16,  sizeof(dt16),  "%Y%m%dT%H%M%SZ", &utc_time);

    
    std::string host         = location.bucket + ".s3." + location.region + ".amazonaws.com";
    std::string object_path  = "/" + location.key;
    std::string payload_hash = sha256_hex("");  

    std::string signed_headers = "host;x-amz-content-sha256;x-amz-date";
    std::string canonical_headers =
        "host:" + host                           + "\n"
        "x-amz-content-sha256:" + payload_hash  + "\n"
        "x-amz-date:" + dt16;

    if (!location.session_token.empty()) {
        canonical_headers += "\nx-amz-security-token:" + location.session_token;
        signed_headers    += ";x-amz-security-token";
    }

    std::string canonical_request =
        "GET\n" + object_path + "\n\n"
        + canonical_headers + "\n\n"
        + signed_headers    + "\n"
        + payload_hash;

    
    std::string credential_scope = std::string(date8) + "/" + location.region + "/s3/aws4_request";
    std::string string_to_sign   =
        "AWS4-HMAC-SHA256\n" + std::string(dt16) + "\n"
        + credential_scope   + "\n"
        + sha256_hex(canonical_request);

    
    std::string signing_key = hmac_sha256_raw(
                                  hmac_sha256_raw(
                                      hmac_sha256_raw(
                                          hmac_sha256_raw("AWS4" + location.secret_key, date8),
                                          location.region),
                                      "s3"),
                                  "aws4_request");

    std::string signature = bytes_to_hex(
        reinterpret_cast<const unsigned char*>(
            hmac_sha256_raw(signing_key, string_to_sign).data()),
        32);

    
    out_auth_header =
        "AWS4-HMAC-SHA256 Credential=" + location.access_key_id + "/" + credential_scope + ","
        + "SignedHeaders=" + signed_headers + ","
        + "Signature=" + signature;

    return "https://" + host + object_path + "?X-Auth=" + out_auth_header;
}
