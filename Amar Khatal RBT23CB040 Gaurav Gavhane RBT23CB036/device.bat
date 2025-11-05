
@echo off
echo Starting ADB device connection setup...

rem Set the port number for ADB
set ADB_PORT=5555

rem Set the path to the ADB executable
set ADB_PATH=adb

rem Check if ADB is available
%ADB_PATH% version >nul 2>&1
if errorlevel 1 (
    echo Error: ADB not found. Please ensure Android SDK platform-tools are installed and in PATH.
    pause
    exit /b 1
)

echo Disconnecting old connections...
%ADB_PATH% disconnect

echo Checking for connected devices...
%ADB_PATH% devices

echo Setting up TCP/IP mode on port %ADB_PORT%...
%ADB_PATH% tcpip %ADB_PORT%

echo Waiting for device to initialize...
timeout /t 3 /nobreak >nul

echo Please ensure your Android device is connected via USB and USB debugging is enabled.
echo Then manually enter your device's IP address when prompted.
set /p DEVICE_IP=Enter your Android device's IP address (e.g., 192.168.1.100): 

if "%DEVICE_IP%"=="" (
    echo No IP address entered. Exiting...
    pause
    exit /b 1
)

echo Connecting to device with IP %DEVICE_IP%:%ADB_PORT%...
%ADB_PATH% connect %DEVICE_IP%:%ADB_PORT%

echo Connection setup complete!
echo You can now disconnect the USB cable and use the device wirelessly.
pause
