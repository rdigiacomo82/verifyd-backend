rule VeriFYD_Suspicious_PowerShell_Download_Execute : verifyd powershell downloader {
    meta:
        verifyd_schema = "1"
        description = "Detects a combination of PowerShell download and execution indicators"
        severity = "high"
        category = "script_downloader"
        confidence = 82
        author = "VeriFYD"
        source = "verifyd_core"
        tags = "powershell,downloader,execution"
        dedupe_key = "powershell_download_execute"
    strings:
        $ps = "powershell" ascii nocase
        $download1 = "Invoke-WebRequest" ascii nocase
        $download2 = "DownloadString" ascii nocase
        $exec1 = "Start-Process" ascii nocase
        $exec2 = "Invoke-Expression" ascii nocase
    condition:
        $ps and 1 of ($download*) and 1 of ($exec*)
}

rule VeriFYD_Encoded_PowerShell_Command : verifyd powershell encoded {
    meta:
        verifyd_schema = "1"
        description = "Detects common PowerShell encoded-command invocation text"
        severity = "medium"
        category = "script_obfuscation"
        confidence = 75
        author = "VeriFYD"
        source = "verifyd_core"
        tags = "powershell,encoded,obfuscation"
        dedupe_key = "powershell_encoded_command"
    strings:
        $ps = "powershell" ascii nocase
        $enc1 = "-EncodedCommand" ascii nocase
        $enc2 = "-enc " ascii nocase
    condition:
        $ps and 1 of ($enc*)
}
