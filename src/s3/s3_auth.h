/**
 * @file s3_auth.h
 * @brief AWS S3 location descriptor and request-signing helpers.
 *
 * Provides S3Loc (parsed S3 URI + credentials), HMAC-SHA256 utilities,
 * path parsing, and URL / Authorization-header generation for the
 * AWS Signature Version 4 ("SigV4") request-signing protocol.
 */
#pragma once

#include <string>

// ─── S3 object location ───────────────────────────────────────────────────────
/**
 * @brief All information needed to authenticate and address one S3 object.
 *
 * Credentials are read from environment variables by parse_s3_path();
 * if no credentials are present the request is treated as anonymous.
 */
struct S3Loc {
    std::string bucket;         ///< S3 bucket name
    std::string key;            ///< Object key (path inside the bucket)
    std::string region;         ///< AWS region (e.g. "us-east-1")
    std::string endpoint;       ///< Host:port endpoint (default: s3.{region}.amazonaws.com)
    std::string access_key_id;  ///< AWS_ACCESS_KEY_ID
    std::string secret_key;     ///< AWS_SECRET_ACCESS_KEY
    std::string session_token;  ///< AWS_SESSION_TOKEN (empty if not using STS)
    bool        is_anonymous = false; ///< True when no credentials are available
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return true if @p path starts with "s3://" or "/vsis3/".
bool is_s3_path(const std::string& path);

/**
 * @brief Parse an S3 URI and populate credentials from environment variables.
 *
 * Accepts both "s3://bucket/key" and "/vsis3/bucket/key" formats.
 * Reads: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
 *        AWS_DEFAULT_REGION, AWS_ENDPOINT_URL.
 */
S3Loc parse_s3_path(const std::string& path);

/**
 * @brief Build an HTTPS URL and populate @p out_auth_header for a GET request.
 *
 * For anonymous locations, returns a plain URL and leaves out_auth_header
 * equal to range_header.  For authenticated locations, generates a SigV4
 * Authorization header and returns the URL with an X-Auth query parameter.
 *
 * @param location        Parsed S3 location (bucket, key, credentials, …).
 * @param range_header    HTTP Range header value (e.g. "bytes=0-1023").
 * @param out_auth_header Output: the Authorization header value to use.
 * @return                The request URL.
 */
std::string build_s3_request_url(const S3Loc&       location,
                                 const std::string& range_header,
                                 std::string&       out_auth_header);
