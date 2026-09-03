; VeriFYD Lens Windows Installer
#define MyAppName "VeriFYD Lens"
#define MyAppVersion "0.4.5"
#define MyAppPublisher "Data By Design LLC"

[Setup]
AppId={{A8E44E8E-8C63-4F30-9DD8-2BC5B733AE9C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\VeriFYD Lens
DefaultGroupName=VeriFYD Lens
OutputDir=installer_output
OutputBaseFilename=VeriFYD_Lens_Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern

[Files]
Source: "VeriFYD_Lens\agent\dist\VeriFYD_Lens_Agent.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "VeriFYD_Lens\agent\dist\VeriFYD_Lens_Activate.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\background.js"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\icon16.png"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\icon32.png"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\icon48.png"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\icon128.png"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\manifest.json"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\popup.html"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\popup.js"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\result.html"; DestDir: "{app}\extension"; Flags: ignoreversion
Source: "VeriFYD_Lens\extension\result.js"; DestDir: "{app}\extension"; Flags: ignoreversion

[Icons]
Name: "{group}\Activate VeriFYD Lens"; Filename: "{app}\VeriFYD_Lens_Activate.exe"
Name: "{group}\VeriFYD Lens Extension Folder"; Filename: "{app}\extension"

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "VeriFYD Lens Agent"; ValueData: """{app}\VeriFYD_Lens_Agent.exe"""; Flags: uninsdeletevalue

[Run]
Filename: "{app}\VeriFYD_Lens_Agent.exe"; Description: "Start VeriFYD Lens Agent"; Flags: nowait postinstall skipifsilent
Filename: "{app}\VeriFYD_Lens_Activate.exe"; Description: "Activate VeriFYD Lens"; Flags: nowait postinstall skipifsilent unchecked

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM VeriFYD_Lens_Agent.exe /F >nul 2>&1"; Flags: runhidden; RunOnceId: "StopLensAgent"


