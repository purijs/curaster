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


/**
 * @brief All information needed to authenticate and address one S3 object.
 *
 * Credentials are read from environment variables by parse_s3_path();
 * if no credentials are present the request is treated as anonymous.
 */
struct S3Loc {
    std::string bucket;
    std::string key;
    std::string region;
    std::string endpoint;
    std::string access_key_id;
    std::string secret_key;
    std::string session_token;
    bool        is_anonymous = false;
};



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
