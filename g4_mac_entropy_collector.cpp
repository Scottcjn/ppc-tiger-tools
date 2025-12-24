/*
 * G4 MAC + Entropy Collector for RustChain PoA
 * Collects hardware signals including MAC addresses for attestation
 *
 * Compile: g++ -O2 -o g4_collector g4_mac_entropy_collector.cpp
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <sys/utsname.h>
#include <sys/sysctl.h>

using namespace std;

// Collect MAC address from en0
string get_mac_en0() {
    FILE* fp = popen("/sbin/ifconfig en0 2>/dev/null | /usr/bin/awk '/ether/{print $2; exit}'", "r");
    if (!fp) return "";

    char buf[128] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        pclose(fp);
        string mac(buf);
        // Trim newline
        while (!mac.empty() && (mac[mac.length()-1] == '\n' || mac[mac.length()-1] == '\r'))
            mac.erase(mac.length()-1);
        return mac;
    }

    pclose(fp);
    return "";
}

// Collect all network MAC addresses
vector<string> get_all_macs() {
    vector<string> macs;

    FILE* fp = popen("/sbin/ifconfig 2>/dev/null | /usr/bin/awk '/ether/{print $2}'", "r");
    if (!fp) return macs;

    char buf[128];
    while (fgets(buf, sizeof(buf), fp)) {
        string mac(buf);
        // Trim newline
        while (!mac.empty() && (mac[mac.length()-1] == '\n' || mac[mac.length()-1] == '\r'))
            mac.erase(mac.length()-1);

        if (!mac.empty())
            macs.push_back(mac);
    }

    pclose(fp);
    return macs;
}

// Collect system entropy (Darwin-specific)
string get_system_entropy() {
    FILE* fp = popen("/usr/bin/ioreg -l 2>/dev/null | /usr/bin/grep -i 'IOPlatformSerialNumber\\|IOPlatformUUID' | head -2", "r");
    if (!fp) return "";

    stringstream ss;
    char buf[256];
    while (fgets(buf, sizeof(buf), fp)) {
        ss << buf;
    }

    pclose(fp);
    return ss.str();
}

// Collect PowerPC-specific data
string get_ppc_info() {
    struct utsname sys_info;
    uname(&sys_info);

    stringstream info;
    info << "machine=" << sys_info.machine << ";";
    info << "sysname=" << sys_info.sysname << ";";
    info << "release=" << sys_info.release << ";";

    // Get CPU info on Darwin PowerPC
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    if (sysctlbyname("hw.model", &cpu_brand, &size, NULL, 0) == 0) {
        info << "hw_model=" << cpu_brand << ";";
    }

    return info.str();
}

// Build JSON attestation payload
string build_attestation_json(const string& miner_id) {
    vector<string> macs = get_all_macs();
    string ppc_info = get_ppc_info();
    string entropy = get_system_entropy();

    struct utsname sys_info;
    uname(&sys_info);

    stringstream json;
    json << "{";
    json << "\"miner\":\"" << miner_id << "\",";

    // Device info
    json << "\"device\":{";
    json << "\"family\":\"PowerPC\",";
    json << "\"arch\":\"G4\",";
    json << "\"machine\":\"" << sys_info.machine << "\",";
    json << "\"sysname\":\"" << sys_info.sysname << "\",";
    json << "\"release\":\"" << sys_info.release << "\"";
    json << "},";

    // Signals
    json << "\"signals\":{";

    // MACs array
    json << "\"macs\":[";
    for (size_t i = 0; i < macs.size(); i++) {
        json << "\"" << macs[i] << "\"";
        if (i < macs.size() - 1) json << ",";
    }
    json << "],";

    // PowerPC info
    json << "\"ppc_info\":\"" << ppc_info << "\",";

    // System entropy sample
    json << "\"entropy_sample\":\"" << entropy << "\"";

    json << "}";  // close signals

    json << "}";  // close root

    return json.str();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <miner_id>" << endl;
        cout << "Example: " << argv[0] << " ppc_g4_98ad7c..." << endl;
        return 1;
    }

    string miner_id = argv[1];

    cout << "================================================================================" << endl;
    cout << "RustChain G4 Hardware Attestation Collector" << endl;
    cout << "================================================================================" << endl;

    // Collect and display info
    vector<string> macs = get_all_macs();
    cout << "MACs found: " << macs.size() << endl;
    for (size_t i = 0; i < macs.size(); i++) {
        cout << "  - " << macs[i] << endl;
    }

    cout << "\nPowerPC Info:" << endl;
    cout << get_ppc_info() << endl;

    cout << "\nBuilding attestation JSON..." << endl;
    string attestation = build_attestation_json(miner_id);

    cout << "\n================================================================================" << endl;
    cout << "ATTESTATION JSON (copy this):" << endl;
    cout << "================================================================================" << endl;
    cout << attestation << endl;
    cout << "================================================================================" << endl;

    return 0;
}
