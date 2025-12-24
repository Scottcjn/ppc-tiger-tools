/*
 * RustChain PowerPC G4 Miner
 * C++ Edition for Mac OS X 10.4 Tiger
 * Compiles with GCC 3.3 or later
 */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

// Network includes for HTTP requests
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>

// Configuration
const std::string SERVER_HOST = "50.28.86.131";
const int SERVER_PORT = 8088;
const std::string SERVER_URL = "http://50.28.86.131:8088";

// Simple SHA-1 implementation for ancient systems
class SimpleSHA1 {
private:
    uint32_t h[5];
    uint32_t buffer[80];
    uint64_t length;

    uint32_t rotateLeft(uint32_t value, int amount) {
        return (value << amount) | (value >> (32 - amount));
    }

    void processBlock() {
        // Extend 16 32-bit words to 80
        for (int i = 16; i < 80; i++) {
            buffer[i] = rotateLeft(buffer[i-3] ^ buffer[i-8] ^ buffer[i-14] ^ buffer[i-16], 1);
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];

        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
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

            uint32_t temp = rotateLeft(a, 5) + f + e + k + buffer[i];
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

    void update(const std::string& data) {
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

    std::string hexdigest() {
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

        std::ostringstream oss;
        oss << std::hex;
        for (int i = 0; i < 5; i++) {
            oss << std::setfill('0') << std::setw(8) << h[i];
        }
        return oss.str();
    }
};

// Hardware info structure
struct HardwareInfo {
    std::string model;
    std::string cpu;
    std::string speed;
    std::string serial;
};

// Execute command and get output
std::string execCommand(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";

    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

// Get hardware information
HardwareInfo getHardwareInfo() {
    HardwareInfo info;

    std::string hwData = execCommand("system_profiler SPHardwareDataType");

    std::istringstream ss(hwData);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("Machine Model:") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                info.model = line.substr(pos + 1);
                // Trim whitespace
                info.model.erase(0, info.model.find_first_not_of(" \t"));
                info.model.erase(info.model.find_last_not_of(" \t") + 1);
            }
        } else if (line.find("CPU Type:") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                info.cpu = line.substr(pos + 1);
                info.cpu.erase(0, info.cpu.find_first_not_of(" \t"));
                info.cpu.erase(info.cpu.find_last_not_of(" \t") + 1);
            }
        } else if (line.find("CPU Speed:") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                info.speed = line.substr(pos + 1);
                info.speed.erase(0, info.speed.find_first_not_of(" \t"));
                info.speed.erase(info.speed.find_last_not_of(" \t") + 1);
            }
        } else if (line.find("Serial Number:") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                info.serial = line.substr(pos + 1);
                info.serial.erase(0, info.serial.find_first_not_of(" \t"));
                info.serial.erase(info.serial.find_last_not_of(" \t") + 1);
            }
        }
    }

    return info;
}

// Collect timing entropy
std::pair<double, double> collectEntropy() {
    std::vector<double> samples;

    for (int i = 0; i < 50; i++) {
        clock_t start = clock();

        // CPU-intensive work for timing variance
        volatile int x = 0;
        for (int j = 0; j < 10000; j++) {
            x ^= (j * i) % 1000;
        }

        clock_t end = clock();
        double duration = ((double)(end - start)) / CLOCKS_PER_SEC * 1000000;
        samples.push_back(duration);
    }

    // Calculate mean
    double sum = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        sum += samples[i];
    }
    double mean = sum / samples.size();

    // Calculate variance
    double varSum = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        double diff = samples[i] - mean;
        varSum += diff * diff;
    }
    double variance = varSum / samples.size();

    return std::make_pair(mean, variance);
}

// Simple HTTP client
std::string httpRequest(const std::string& host, int port, const std::string& path,
                       const std::string& method = "GET", const std::string& data = "") {

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

    std::ostringstream request;
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

    std::string req = request.str();
    send(sockfd, req.c_str(), req.length(), 0);

    char buffer[4096];
    std::string response;
    int bytes;
    while ((bytes = recv(sockfd, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes] = '\0';
        response += buffer;
    }

    close(sockfd);

    // Extract body (after \r\n\r\n)
    size_t bodyStart = response.find("\r\n\r\n");
    if (bodyStart != std::string::npos) {
        return response.substr(bodyStart + 4);
    }

    return response;
}

// Extract value from JSON-like response
std::string extractJsonValue(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\":\"";
    size_t start = json.find(searchKey);
    if (start == std::string::npos) return "";

    start += searchKey.length();
    size_t end = json.find("\"", start);
    if (end == std::string::npos) return "";

    return json.substr(start, end - start);
}

// Extract numeric value from JSON
double extractJsonNumber(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\":";
    size_t start = json.find(searchKey);
    if (start == std::string::npos) return 0.0;

    start += searchKey.length();
    size_t end = json.find_first_of(",}", start);
    if (end == std::string::npos) return 0.0;

    std::string value = json.substr(start, end - start);
    return atof(value.c_str());
}

int main() {
    std::cout << "RustChain PowerPC G4 Miner - C++ Edition" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Server: " << SERVER_URL << std::endl;
    std::cout << std::endl;

    // Get hardware info
    std::cout << "Detecting PowerPC G4 hardware..." << std::endl;
    HardwareInfo hw = getHardwareInfo();

    std::cout << "Model: " << (hw.model.empty() ? "PowerBook G4" : hw.model) << std::endl;
    std::cout << "CPU: " << (hw.cpu.empty() ? "PowerPC G4" : hw.cpu) << std::endl;
    std::cout << "Speed: " << (hw.speed.empty() ? "1.5 GHz" : hw.speed) << std::endl;
    std::cout << "Serial: " << hw.serial.substr(0, 8) << "..." << std::endl;
    std::cout << std::endl;

    // Create system ID
    SimpleSHA1 sha;
    sha.update(hw.serial.empty() ? "G4-FALLBACK" : hw.serial);
    std::string systemId = "g4_" + sha.hexdigest().substr(0, 12);
    std::cout << "System ID: " << systemId << std::endl;
    std::cout << std::endl;

    // Test connectivity
    std::cout << "Testing server connectivity..." << std::endl;
    std::string statsResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/api/stats");
    if (statsResponse.find("version") == std::string::npos) {
        std::cout << "ERROR: Cannot reach RustChain server!" << std::endl;
        return 1;
    }
    std::cout << "Server reachable: RustChain v2" << std::endl;
    std::cout << std::endl;

    // Collect entropy
    std::cout << "Collecting entropy..." << std::endl;
    std::pair<double, double> entropy = collectEntropy();
    std::cout << "CPU timing mean: " << entropy.first << " microseconds" << std::endl;
    std::cout << "CPU timing variance: " << entropy.second << std::endl;
    std::cout << std::endl;

    // Get challenge
    std::cout << "Getting attestation challenge..." << std::endl;
    std::string challengeResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/attest/challenge", "POST", "{}");
    std::string nonce = extractJsonValue(challengeResponse, "nonce");

    if (nonce.empty()) {
        std::cout << "ERROR: Failed to get challenge nonce" << std::endl;
        return 1;
    }

    std::cout << "Challenge nonce: " << nonce.substr(0, 8) << "..." << std::endl;

    // Create commitment
    SimpleSHA1 commitSha;
    std::ostringstream commitData;
    commitData << entropy.first << entropy.second << nonce;
    commitSha.update(commitData.str());
    std::string commitment = "b3:" + commitSha.hexdigest();

    // Build attestation
    std::ostringstream attestation;
    attestation << "{";
    attestation << "\"report\":{";
    attestation << "\"nonce\":\"" << nonce << "\",";
    attestation << "\"device\":{\"family\":\"PowerPC\",\"arch\":\"G4\",\"model\":\"" << hw.model << "\",\"year\":2005},";
    attestation << "\"derived\":{\"cpu_drift_mean\":" << entropy.first << ",\"cpu_drift_var\":" << entropy.second << "},";
    attestation << "\"commitment\":\"" << commitment << "\",";
    attestation << "\"timestamp\":" << time(NULL);
    attestation << "},";
    attestation << "\"miner_pubkey\":\"" << systemId << "\",";
    attestation << "\"miner_sig\":\"\"";
    attestation << "}";

    // Submit attestation
    std::cout << "Submitting Silicon Ticket attestation..." << std::endl;
    std::string attestResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/attest/submit", "POST", attestation.str());
    std::string ticketId = extractJsonValue(attestResponse, "ticket_id");

    if (ticketId.empty()) {
        std::cout << "ERROR: Attestation failed!" << std::endl;
        std::cout << "Response: " << attestResponse.substr(0, 200) << "..." << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Got Silicon Ticket: " << ticketId << std::endl;
    std::cout << std::endl;

    // Enroll in epoch
    std::cout << "Enrolling in current epoch..." << std::endl;
    std::ostringstream enrollment;
    enrollment << "{";
    enrollment << "\"miner_pubkey\":\"" << systemId << "\",";
    enrollment << "\"weights\":{\"temporal\":1.0,\"rtc\":1.0},";
    enrollment << "\"device\":{\"family\":\"PowerPC\",\"arch\":\"G4\",\"model\":\"" << hw.model << "\",\"year\":2005},";
    enrollment << "\"ticket_id\":\"" << ticketId << "\",";
    enrollment << "\"slot\":" << (time(NULL) / 600);
    enrollment << "}";

    std::string enrollResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/epoch/enroll", "POST", enrollment.str());

    if (enrollResponse.find("\"ok\":true") == std::string::npos) {
        std::cout << "ERROR: Epoch enrollment failed!" << std::endl;
        std::cout << "Response: " << enrollResponse.substr(0, 200) << "..." << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Enrolled in epoch!" << std::endl;
    std::cout << "Hardware multiplier: 2.5x (Classic PowerPC G4)" << std::endl;
    std::cout << std::endl;

    // Check balance
    std::string balanceResponse = httpRequest(SERVER_HOST, SERVER_PORT, "/balance/" + systemId);
    double balance = extractJsonNumber(balanceResponse, "balance_rtc");

    std::cout << "Current balance: " << std::fixed << balance << " RTC" << std::endl;
    std::cout << std::endl;

    std::cout << "PowerPC G4 is now mining RustChain!" << std::endl;
    std::cout << "- Epoch-based pro-rata rewards" << std::endl;
    std::cout << "- 2.5x Classic hardware advantage" << std::endl;
    std::cout << "- Rewards distributed every 24 hours" << std::endl;
    std::cout << std::endl;
    std::cout << "Monitor balance at:" << std::endl;
    std::cout << SERVER_URL << "/balance/" << systemId << std::endl;
    std::cout << std::endl;
    std::cout << "Press Ctrl+C to stop monitoring..." << std::endl;

    // Balance monitoring loop
    int count = 0;
    while (true) {
        sleep(300); // 5 minutes

        std::string balResp = httpRequest(SERVER_HOST, SERVER_PORT, "/balance/" + systemId);
        double bal = extractJsonNumber(balResp, "balance_rtc");

        count++;
        std::cout << "[" << count << "] Balance: " << std::fixed << bal << " RTC" << std::endl;
    }

    return 0;
}