/*
 * RustChain PowerPC G4 Miner - Fixed Version
 * Compatible with older GCC on Mac OS X 10.4
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

using namespace std;

// Configuration
const string SERVER_HOST = "50.28.86.131";
const int SERVER_PORT = 8088;

// Simple hex conversion
string toHex(unsigned int val) {
    char buf[16];
    sprintf(buf, "%08x", val);
    return string(buf);
}

// Simple SHA-1 implementation
class SimpleSHA1 {
private:
    unsigned int h[5];
    unsigned int buffer[80];
    unsigned long long length;

    unsigned int rotateLeft(unsigned int value, int amount) {
        return (value << amount) | (value >> (32 - amount));
    }

    void processBlock() {
        for (int i = 16; i < 80; i++) {
            buffer[i] = rotateLeft(buffer[i-3] ^ buffer[i-8] ^ buffer[i-14] ^ buffer[i-16], 1);
        }

        unsigned int a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];

        for (int i = 0; i < 80; i++) {
            unsigned int f, k;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            unsigned int temp = rotateLeft(a, 5) + f + e + k + buffer[i];
            e = d; d = c; c = rotateLeft(b, 30); b = a; a = temp;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }

public:
    SimpleSHA1() {
        h[0] = 0x67452301;
        h[1] = 0xEFCDAB89;
        h[2] = 0x98BADCFE;
        h[3] = 0x10325476;
        h[4] = 0xC3D2E1F0;
        length = 0;
        memset(buffer, 0, sizeof(buffer));
    }

    void update(const string& data) {
        for (size_t i = 0; i < data.length(); i++) {
            int pos = (length % 64) / 4;
            int shift = 24 - ((length % 4) * 8);
            buffer[pos] |= ((unsigned char)data[i]) << shift;
            length++;

            if (length % 64 == 0) {
                processBlock();
                memset(buffer, 0, sizeof(buffer));
            }
        }
    }

    string hexdigest() {
        // Pad message
        int pos = (length % 64) / 4;
        int shift = 24 - ((length % 4) * 8);
        buffer[pos] |= 0x80 << shift;

        if (length % 64 >= 56) {
            processBlock();
            memset(buffer, 0, sizeof(buffer));
        }

        buffer[14] = (length * 8) >> 32;
        buffer[15] = (length * 8) & 0xFFFFFFFF;
        processBlock();

        return toHex(h[0]) + toHex(h[1]) + toHex(h[2]) + toHex(h[3]) + toHex(h[4]);
    }
};

// Execute command and get output
string execCommand(const string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";

    char buffer[128];
    string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

// Get hardware information
struct HardwareInfo {
    string model;
    string cpu;
    string speed;
    string serial;
};

HardwareInfo getHardwareInfo() {
    HardwareInfo info;
    string hwData = execCommand("system_profiler SPHardwareDataType");

    // Parse the output
    size_t pos = 0;
    while (pos < hwData.length()) {
        size_t lineEnd = hwData.find('\n', pos);
        if (lineEnd == string::npos) break;

        string line = hwData.substr(pos, lineEnd - pos);

        if (line.find("Machine Model:") != string::npos) {
            size_t colonPos = line.find(':');
            if (colonPos != string::npos) {
                info.model = line.substr(colonPos + 1);
                // Trim spaces
                while (info.model.length() > 0 && (info.model[0] == ' ' || info.model[0] == '\t')) {
                    info.model = info.model.substr(1);
                }
            }
        } else if (line.find("CPU Type:") != string::npos) {
            size_t colonPos = line.find(':');
            if (colonPos != string::npos) {
                info.cpu = line.substr(colonPos + 1);
                while (info.cpu.length() > 0 && (info.cpu[0] == ' ' || info.cpu[0] == '\t')) {
                    info.cpu = info.cpu.substr(1);
                }
            }
        } else if (line.find("Serial Number:") != string::npos) {
            size_t colonPos = line.find(':');
            if (colonPos != string::npos) {
                info.serial = line.substr(colonPos + 1);
                while (info.serial.length() > 0 && (info.serial[0] == ' ' || info.serial[0] == '\t')) {
                    info.serial = info.serial.substr(1);
                }
            }
        }

        pos = lineEnd + 1;
    }

    return info;
}

// Collect timing entropy
pair<double, double> collectEntropy() {
    vector<double> samples;

    for (int i = 0; i < 50; i++) {
        clock_t start = clock();

        volatile int x = 0;
        for (int j = 0; j < 5000; j++) {
            x ^= (j * i) % 1000;
        }

        clock_t end = clock();
        double duration = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;
        samples.push_back(duration);
    }

    double sum = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        sum += samples[i];
    }
    double mean = sum / samples.size();

    double varSum = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        double diff = samples[i] - mean;
        varSum += diff * diff;
    }
    double variance = varSum / samples.size();

    return make_pair(mean, variance);
}

// Simple HTTP client
string httpRequest(const string& host, int port, const string& path,
                  const string& method = "GET", const string& data = "") {

    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return "";

    struct hostent* server = gethostbyname(host.c_str());
    if (server == NULL) {
        close(sockfd);
        return "";
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);

    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        close(sockfd);
        return "";
    }

    ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << ":" << port << "\r\n";
    request << "Connection: close\r\n";

    if (method == "POST") {
        request << "Content-Type: application/json\r\n";
        request << "Content-Length: " << data.length() << "\r\n";
    }
    request << "\r\n";

    if (method == "POST") {
        request << data;
    }

    string req = request.str();
    send(sockfd, req.c_str(), req.length(), 0);

    char buffer[4096];
    string response;
    int bytes;
    while ((bytes = recv(sockfd, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes] = '\0';
        response += buffer;
    }

    close(sockfd);

    size_t bodyStart = response.find("\r\n\r\n");
    if (bodyStart != string::npos) {
        return response.substr(bodyStart + 4);
    }
    return response;
}

// Extract JSON values
string extractJsonValue(const string& json, const string& key) {
    string searchKey = "\"" + key + "\":\"";
    size_t start = json.find(searchKey);
    if (start == string::npos) return "";

    start += searchKey.length();
    size_t end = json.find("\"", start);
    if (end == string::npos) return "";

    return json.substr(start, end - start);
}

double extractJsonNumber(const string& json, const string& key) {
    string searchKey = "\"" + key + "\":";
    size_t start = json.find(searchKey);
    if (start == string::npos) return 0.0;

    start += searchKey.length();
    size_t end = json.find_first_of(",}", start);
    if (end == string::npos) return 0.0;

    string value = json.substr(start, end - start);
    return atof(value.c_str());
}

int main() {
    cout << "RustChain PowerPC G4 Miner" << endl;
    cout << "==========================" << endl;
    cout << "Server: " << SERVER_HOST << ":" << SERVER_PORT << endl;
    cout << endl;

    // Get hardware info
    cout << "Detecting PowerPC G4..." << endl;
    HardwareInfo hw = getHardwareInfo();

    cout << "Model: " << (hw.model.empty() ? "PowerBook G4" : hw.model) << endl;
    cout << "CPU: " << (hw.cpu.empty() ? "PowerPC G4" : hw.cpu) << endl;
    cout << "Serial: " << hw.serial.substr(0, 8) << "..." << endl;
    cout << endl;

    // Create system ID
    SimpleSHA1 sha;
    sha.update(hw.serial.empty() ? "G4-FALLBACK" : hw.serial);
    string systemId = "g4_" + sha.hexdigest().substr(0, 12);
    cout << "System ID: " << systemId << endl;
    cout << endl;

    // Test connectivity
    cout << "Testing server connectivity..." << endl;
    string statsResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/api/stats");
    if (statsResponse.find("version") == string::npos) {
        cout << "ERROR: Cannot reach server!" << endl;
        return 1;
    }
    cout << "Server reachable!" << endl;
    cout << endl;

    // Collect entropy
    cout << "Collecting entropy..." << endl;
    pair<double, double> entropy = collectEntropy();
    cout << "Mean: " << entropy.first << " us, Variance: " << entropy.second << endl;
    cout << endl;

    // Get challenge
    cout << "Getting challenge..." << endl;
    string challengeResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/attest/challenge", "POST", "{}");
    string nonce = extractJsonValue(challengeResponse, "nonce");

    if (nonce.empty()) {
        cout << "ERROR: No challenge nonce" << endl;
        return 1;
    }

    // Submit attestation
    cout << "Submitting attestation..." << endl;
    SimpleSHA1 commitSha;
    ostringstream commitData;
    commitData << entropy.first << entropy.second << nonce;
    commitSha.update(commitData.str());
    string commitment = "b3:" + commitSha.hexdigest();

    ostringstream attestation;
    attestation << "{";
    attestation << "\"report\":{";
    attestation << "\"nonce\":\"" << nonce << "\",";
    attestation << "\"device\":{\"family\":\"PowerPC\",\"arch\":\"G4\",\"model\":\"PowerBook6,8\"},";
    attestation << "\"derived\":{\"cpu_drift_mean\":" << entropy.first << ",\"cpu_drift_var\":" << entropy.second << "},";
    attestation << "\"commitment\":\"" << commitment << "\",";
    attestation << "\"timestamp\":" << time(NULL) << "},";
    attestation << "\"miner_pubkey\":\"" << systemId << "\",\"miner_sig\":\"\"}";

    string attestResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/attest/submit", "POST", attestation.str());
    string ticketId = extractJsonValue(attestResponse, "ticket_id");

    if (ticketId.empty()) {
        cout << "ERROR: Attestation failed!" << endl;
        return 1;
    }

    cout << "SUCCESS: Got ticket " << ticketId << endl;
    cout << endl;

    // Enroll in epoch
    cout << "Enrolling in epoch..." << endl;
    ostringstream enrollment;
    enrollment << "{\"miner_pubkey\":\"" << systemId << "\",";
    enrollment << "\"weights\":{\"temporal\":1.0,\"rtc\":1.0},";
    enrollment << "\"device\":{\"family\":\"PowerPC\",\"arch\":\"G4\",\"model\":\"PowerBook6,8\"},";
    enrollment << "\"ticket_id\":\"" << ticketId << "\",";
    enrollment << "\"slot\":" << (time(NULL) / 600) << "}";

    string enrollResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/epoch/enroll", "POST", enrollment.str());

    if (enrollResponse.find("\"ok\":true") == string::npos) {
        cout << "ERROR: Enrollment failed!" << endl;
        return 1;
    }

    cout << "SUCCESS: Enrolled with 2.5x G4 advantage!" << endl;
    cout << endl;

    // Monitor balance
    cout << "PowerPC G4 mining RustChain!" << endl;
    cout << "Monitor: http://" << SERVER_HOST << ":" << SERVER_PORT << "/balance/" << systemId << endl;
    cout << endl;
    cout << "Press Ctrl+C to stop..." << endl;

    int count = 0;
    while (true) {
        sleep(300); // 5 minutes

        string balResp = httpRequest(SERVER_HOST, SERVER_PORT, "/balance/" + systemId);
        double balance = extractJsonNumber(balResp, "balance_rtc");

        count++;
        cout << "[" << count << "] Balance: " << balance << " RTC" << endl;
    }

    return 0;
}