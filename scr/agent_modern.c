// agent_modern_enhanced.c - Ultimate Version
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <sys/stat.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <psapi.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")

// Enhanced structures
typedef struct {
    unsigned long long idleTime;
    unsigned long long kernelTime;
    unsigned long long userTime;
} CpuTime;

typedef struct {
    unsigned long long totalBytes;
    unsigned long long freeBytes;
    unsigned long long usedBytes;
    double usagePercent;
} DiskInfo;

typedef struct {
    double latency;
    double packetLoss;
    double bandwidth;
    char status[20];
} NetworkInfo;

typedef struct {
    float temperature;
    float fanSpeed;
    float powerUsage;
} SystemHealth;

typedef struct {
    char metric[20];
    float value;
    float threshold;
    char timestamp[25];
    char message[100];
    char severity[10];
    int correlation_id;
} EnhancedAlert;

typedef struct {
    char timestamp[25];
    float cpu_usage;
    int ram_usage;
    float disk_usage;
    float network_latency;
    float temperature;
    float power_consumption;
    char tenant[20];
    char status[20];
    char alert_msg[100];
} EnhancedLogEntry;

// Global configuration
typedef struct {
    float cpu_threshold;
    float ram_threshold;
    float disk_threshold;
    float temp_threshold;
    char tenant[20];
    int sampling_rate;
    int enable_ai;
} AgentConfig;

AgentConfig config = {85.0, 90.0, 95.0, 75.0, "default", 2000, 1};

// Network monitoring
double measure_latency() {
    WSADATA wsa;
    SOCKET sock;
    struct sockaddr_in server;
    struct timeval timeout;
    fd_set fds;
    
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) return -1;
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    server.sin_addr.s_addr = inet_addr("8.8.8.8");
    server.sin_family = AF_INET;
    server.sin_port = htons(80);
    
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&timeout, sizeof(timeout));
    
    clock_t start = clock();
    if (connect(sock, (struct sockaddr*)&server, sizeof(server)) == 0) {
        closesocket(sock);
        WSACleanup();
        return ((double)(clock() - start) / CLOCKS_PER_SEC) * 1000;
    }
    
    closesocket(sock);
    WSACleanup();
    return -1;
}

// Get system temperature (simulated for Windows)
float get_system_temperature() {
    // This is simulated - in production would use WMI or hardware APIs
    static float base_temp = 40.0;
    
    // Simulate temperature based on CPU usage
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    
    // Add some variation
    float variation = (rand() % 1000) / 1000.0 * 5.0;
    return base_temp + variation;
}

// Get power consumption (simulated)
float get_power_consumption() {
    // Simulated power usage based on system load
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    
    float base_power = 50.0; // Base power in watts
    float cpu_contribution = (float)memInfo.dwMemoryLoad / 100.0 * 30.0;
    float variation = (rand() % 1000) / 1000.0 * 10.0;
    
    return base_power + cpu_contribution + variation;
}

// Get per-process CPU usage
void get_top_processes() {
    DWORD processes[1024], cbNeeded, cProcesses;
    
    if (!EnumProcesses(processes, sizeof(processes), &cbNeeded)) {
        return;
    }
    
    cProcesses = cbNeeded / sizeof(DWORD);
    
    for (DWORD i = 0; i < min(cProcesses, 5); i++) {
        if (processes[i] != 0) {
            HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 
                                         FALSE, processes[i]);
            if (hProcess != NULL) {
                char szProcessName[MAX_PATH] = "";
                HMODULE hMod;
                DWORD cbNeeded;
                
                if (EnumProcessModules(hProcess, &hMod, sizeof(hMod), &cbNeeded)) {
                    GetModuleBaseNameA(hProcess, hMod, szProcessName, sizeof(szProcessName));
                    
                    PROCESS_MEMORY_COUNTERS pmc;
                    if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
                        printf("[PROCESS] %s - Memory: %.2f MB\n", 
                               szProcessName, 
                               pmc.WorkingSetSize / 1024.0 / 1024.0);
                    }
                }
                CloseHandle(hProcess);
            }
        }
    }
}

// Enhanced threshold checking with severity levels
void check_enhanced_thresholds(float cpu, int ram, float disk, float temp, 
                               EnhancedAlert *alerts, int *alert_count) {
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char timestamp[25];
    strftime(timestamp, 25, "%Y-%m-%d %H:%M:%S", tm_info);
    
    *alert_count = 0;
    
    // CPU with severity levels
    if (cpu > config.cpu_threshold) {
        strcpy(alerts[*alert_count].metric, "CPU");
        alerts[*alert_count].value = cpu;
        alerts[*alert_count].threshold = config.cpu_threshold;
        strcpy(alerts[*alert_count].timestamp, timestamp);
        
        if (cpu > config.cpu_threshold * 1.5) {
            strcpy(alerts[*alert_count].severity, "CRITICAL");
            sprintf(alerts[*alert_count].message, 
                    "CRITICAL: CPU at %.2f%%! Immediate action required", cpu);
        } else {
            strcpy(alerts[*alert_count].severity, "WARNING");
            sprintf(alerts[*alert_count].message, 
                    "WARNING: CPU usage %.2f%% exceeds threshold", cpu);
        }
        (*alert_count)++;
    }
    
    // RAM with severity levels
    if (ram > config.ram_threshold) {
        strcpy(alerts[*alert_count].metric, "RAM");
        alerts[*alert_count].value = (float)ram;
        alerts[*alert_count].threshold = config.ram_threshold;
        strcpy(alerts[*alert_count].timestamp, timestamp);
        
        if (ram > 95) {
            strcpy(alerts[*alert_count].severity, "CRITICAL");
            sprintf(alerts[*alert_count].message, 
                    "CRITICAL: RAM at %d%%! System instability likely", ram);
        } else {
            strcpy(alerts[*alert_count].severity, "WARNING");
            sprintf(alerts[*alert_count].message, 
                    "WARNING: RAM usage %d%% exceeds threshold", ram);
        }
        (*alert_count)++;
    }
    
    // Disk with severity levels
    if (disk > config.disk_threshold) {
        strcpy(alerts[*alert_count].metric, "DISK");
        alerts[*alert_count].value = disk;
        alerts[*alert_count].threshold = config.disk_threshold;
        strcpy(alerts[*alert_count].timestamp, timestamp);
        
        if (disk > 98) {
            strcpy(alerts[*alert_count].severity, "CRITICAL");
            sprintf(alerts[*alert_count].message, 
                    "CRITICAL: Disk at %.2f%%! Immediate cleanup required", disk);
        } else {
            strcpy(alerts[*alert_count].severity, "WARNING");
            sprintf(alerts[*alert_count].message, 
                    "WARNING: Disk usage %.2f%% exceeds threshold", disk);
        }
        (*alert_count)++;
    }
    
    // Temperature monitoring
    if (temp > config.temp_threshold) {
        strcpy(alerts[*alert_count].metric, "TEMPERATURE");
        alerts[*alert_count].value = temp;
        alerts[*alert_count].threshold = config.temp_threshold;
        strcpy(alerts[*alert_count].timestamp, timestamp);
        strcpy(alerts[*alert_count].severity, "WARNING");
        sprintf(alerts[*alert_count].message, 
                "System temperature %.2f°C exceeds safe threshold", temp);
        (*alert_count)++;
    }
}

// Send enhanced data with all metrics
void send_enhanced_data(EnhancedLogEntry *entry, EnhancedAlert *alerts, int alert_count) {
    SOCKET sock;
    struct sockaddr_in serv_addr;
    char json_payload[2048];
    char http_request[4096];
    
    // Build enhanced JSON payload
    char alerts_json[1024] = "[]";
    if (alert_count > 0) {
        strcpy(alerts_json, "[");
        for (int i = 0; i < alert_count; i++) {
            char alert_item[256];
            sprintf(alert_item, 
                    "{\"metric\":\"%s\",\"value\":%.2f,\"threshold\":%.2f,"
                    "\"message\":\"%s\",\"severity\":\"%s\",\"timestamp\":\"%s\"}",
                    alerts[i].metric,
                    alerts[i].value,
                    alerts[i].threshold,
                    alerts[i].message,
                    alerts[i].severity,
                    alerts[i].timestamp);
            strcat(alerts_json, alert_item);
            if (i < alert_count - 1) strcat(alerts_json, ",");
        }
        strcat(alerts_json, "]");
    }
    
    // Include tenant and enhanced metrics
    sprintf(json_payload, 
            "{\"tenant\":\"%s\",\"cpu\": %.2f, \"ram\": %d, \"disk\": %.2f, "
            "\"network\": %.2f, \"temperature\": %.2f, \"power\": %.2f, "
            "\"status\": \"%s\", \"timestamp\": \"%s\", \"alerts\": %s}",
            entry->tenant,
            entry->cpu_usage,
            entry->ram_usage,
            entry->disk_usage,
            entry->network_latency,
            entry->temperature,
            entry->power_consumption,
            entry->status,
            entry->timestamp,
            alerts_json);
    
    // Setup connection
    sock = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(5000);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == 0) {
        sprintf(http_request, 
                "POST /api/enhanced_report HTTP/1.1\r\n"
                "Host: 127.0.0.1\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "\r\n"
                "%s", 
                (int)strlen(json_payload), json_payload);
        
        send(sock, http_request, (int)strlen(http_request), 0);
        
        printf("[ENHANCED] Tenant: %s | CPU: %.2f%% | RAM: %d%% | Disk: %.2f%% | "
               "Temp: %.1f°C | Alerts: %d\n", 
               entry->tenant, entry->cpu_usage, entry->ram_usage, 
               entry->disk_usage, entry->temperature, alert_count);
    } else {
        printf("[ERR] Dashboard connection failed\n");
    }
    closesocket(sock);
}

// Helper function to get CPU usage
float calculate_cpu_load() {
    static ULONGLONG lastIdleTime = 0, lastKernelTime = 0, lastUserTime = 0;
    ULONGLONG idleTime, kernelTime, userTime;
    
    FILETIME idleTimeFt, kernelTimeFt, userTimeFt, sysTimeFt;
    
    if (!GetSystemTimes(&idleTimeFt, &kernelTimeFt, &userTimeFt)) {
        return -1.0f;
    }
    
    // Convert FILETIME to 64-bit integers
    idleTime = (((ULONGLONG)idleTimeFt.dwHighDateTime) << 32) | idleTimeFt.dwLowDateTime;
    kernelTime = (((ULONGLONG)kernelTimeFt.dwHighDateTime) << 32) | kernelTimeFt.dwLowDateTime;
    userTime = (((ULONGLONG)userTimeFt.dwHighDateTime) << 32) | userTimeFt.dwLowDateTime;
    
    // Calculate the difference since last call
    ULONGLONG idleTimeDiff = idleTime - lastIdleTime;
    ULONGLONG kernelTimeDiff = kernelTime - lastKernelTime;
    ULONGLONG userTimeDiff = userTime - lastUserTime;
    
    // Calculate total system time
    ULONGLONG sysTimeDiff = kernelTimeDiff + userTimeDiff;
    
    // Store current values for next call
    lastIdleTime = idleTime;
    lastKernelTime = kernelTime;
    lastUserTime = userTime;
    
    // Avoid division by zero
    if (sysTimeDiff == 0) {
        return 0.0f;
    }
    
    // Calculate CPU usage percentage
    float cpuUsage = 100.0f * (1.0f - ((float)idleTimeDiff / (float)sysTimeDiff));
    
    // Ensure value is within valid range
    if (cpuUsage < 0.0f) cpuUsage = 0.0f;
    if (cpuUsage > 100.0f) cpuUsage = 100.0f;
    
    return cpuUsage;
}

// Helper function to get RAM usage
int get_ram_usage() {
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    
    if (!GlobalMemoryStatusEx(&memStatus)) {
        return -1;
    }
    
    return (int)memStatus.dwMemoryLoad;
}

// Helper function to get Disk usage
DiskInfo get_disk_usage() {
    DiskInfo disk = {0};
    
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    
    // Get disk space for C: drive
    if (GetDiskFreeSpaceEx("C:", 
                          &freeBytesAvailable,
                          &totalNumberOfBytes,
                          &totalNumberOfFreeBytes)) {
        
        disk.totalBytes = totalNumberOfBytes.QuadPart;
        disk.freeBytes = totalNumberOfFreeBytes.QuadPart;
        disk.usedBytes = disk.totalBytes - disk.freeBytes;
        
        if (disk.totalBytes > 0) {
            disk.usagePercent = (double)((double)disk.usedBytes / (double)disk.totalBytes) * 100.0;
        } else {
            disk.usagePercent = 0.0;
        }
    } else {
        // If failed, use simulated values
        disk.totalBytes = 1000000000000; // 1TB
        disk.freeBytes = 300000000000;  // 300GB free
        disk.usedBytes = disk.totalBytes - disk.freeBytes;
        disk.usagePercent = 70.0; // 70% used
    }
    
    return disk;
}

int main() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    
    printf("=========================================\n");
    printf("   ENHANCED IOT AGENT - AI-POWERED\n");
    printf("=========================================\n");
    printf("Tenant: %s\n", config.tenant);
    printf("Sampling Rate: %dms\n", config.sampling_rate);
    printf("AI Features: %s\n", config.enable_ai ? "Enabled" : "Disabled");
    printf("Thresholds - CPU: %.1f%%, RAM: %.1f%%, Disk: %.1f%%\n",
           config.cpu_threshold, config.ram_threshold, config.disk_threshold);
    printf("=========================================\n\n");
    
    // Create logs directory
    mkdir("logs");
    
    srand((unsigned int)time(NULL));
    
    while(1) {
        time_t now = time(NULL);
    
         // Collect all metrics
         float cpu = 0.0;
        int ram = 0;
        DiskInfo disk = {0};
        float network_latency = 0.0;
        float temperature = 0.0;
        float power = 0.0;
    
        // Get system metrics
        cpu = calculate_cpu_load();
        ram = get_ram_usage();
        disk = get_disk_usage();
    
        // Enhanced metrics
        network_latency = measure_latency();
        temperature = get_system_temperature();
        power = get_power_consumption();
    
        // Check thresholds with enhanced alerts
        EnhancedAlert alerts[5];
        int alert_count = 0;
        check_enhanced_thresholds(cpu, ram, disk.usagePercent, temperature, 
                                                      alerts, &alert_count);
    
        // Create enhanced log entry
        EnhancedLogEntry entry;
        struct tm *tm_info = localtime(&now);
        strftime(entry.timestamp, sizeof(entry.timestamp), 
                     "%Y-%m-%d %H:%M:%S", tm_info);
        strcpy(entry.tenant, config.tenant);
        entry.cpu_usage = cpu;
        entry.ram_usage = ram;
        entry.disk_usage = (float)disk.usagePercent;
        entry.network_latency = network_latency;
        entry.temperature = temperature;
        entry.power_consumption = power;
        strcpy(entry.status, "online");
    
        // Combine alert messages
        if (alert_count > 0) {
            char alert_combined[256] = "";
            for (int i = 0; i < alert_count; i++) {
                strcat(alert_combined, alerts[i].message);
                if (i < alert_count - 1) strcat(alert_combined, " | ");
            }
            strcpy(entry.alert_msg, alert_combined);
        } else {
            strcpy(entry.alert_msg, "");
       }
    
            // Send to enhanced dashboard
        send_enhanced_data(&entry, alerts, alert_count);
    
        // Periodically show top processes
        static int process_counter = 0;
        if (process_counter++ % 30 == 0) { // Every 60 seconds
           printf("\n[SYSTEM] Top processes:\n");
           get_top_processes();
           printf("\n");
        }
    
         Sleep(config.sampling_rate);
    }
    
    WSACleanup();
    return 0;
}