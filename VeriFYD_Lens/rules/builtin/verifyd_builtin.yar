/*
VeriFYD Lens Built-in YARA-X Rules
VERIFYD_LENS_YARA_RULES_V1

Conservative Phase 2A starter pack. These are intentionally high-confidence
patterns for download triage. Large third-party rule packs should be added in
Phase 2B only after false-positive review.
*/

rule Verifyd_Phase2_Test {
    meta:
        description = "VeriFYD harmless YARA-X smoke-test rule"
        severity = "LOW"
        category = "test"
    strings:
        $test = "VERIFYD_PHASE2_TEST" ascii wide
    condition:
        $test
}

rule Verifyd_Suspicious_PowerShell_Downloader {
    meta:
        description = "PowerShell downloader keywords commonly used by script droppers"
        severity = "MEDIUM"
        category = "script"
    strings:
        $ps1 = "powershell" nocase ascii wide
        $a = "DownloadString" nocase ascii wide
        $b = "System.Net.WebClient" nocase ascii wide
        $c = "Invoke-WebRequest" nocase ascii wide
        $d = "Start-BitsTransfer" nocase ascii wide
    condition:
        $ps1 and 1 of ($a,$b,$c,$d)
}

rule Verifyd_PowerShell_EncodedCommand {
    meta:
        description = "PowerShell EncodedCommand usage"
        severity = "MEDIUM"
        category = "script"
    strings:
        $ps1 = "powershell" nocase ascii wide
        $e1 = "-EncodedCommand" nocase ascii wide
        $e2 = " -enc " nocase ascii wide
        $e3 = " /enc " nocase ascii wide
    condition:
        $ps1 and 1 of ($e*)
}

rule Verifyd_InvokeExpression_Execution {
    meta:
        description = "Invoke-Expression execution pattern"
        severity = "MEDIUM"
        category = "script"
    strings:
        $a = "Invoke-Expression" nocase ascii wide
        $b = "IEX" ascii wide
    condition:
        $a or $b
}

rule Verifyd_Certutil_Download_Abuse {
    meta:
        description = "certutil download/cache abuse pattern"
        severity = "MEDIUM"
        category = "lolbin"
    strings:
        $c = "certutil" nocase ascii wide
        $u = "-urlcache" nocase ascii wide
        $f = "-f" nocase ascii wide
    condition:
        $c and $u and $f
}

rule Verifyd_Mshta_Execution {
    meta:
        description = "mshta execution pattern"
        severity = "MEDIUM"
        category = "lolbin"
    strings:
        $a = "mshta" nocase ascii wide
        $b = "javascript:" nocase ascii wide
        $c = "vbscript:" nocase ascii wide
    condition:
        $a and any of ($b,$c)
}

rule Verifyd_Rundll32_Javascript {
    meta:
        description = "rundll32 JavaScript execution pattern"
        severity = "MEDIUM"
        category = "lolbin"
    strings:
        $r = "rundll32" nocase ascii wide
        $j = "javascript:" nocase ascii wide
        $m = "mshtml" nocase ascii wide
    condition:
        $r and $j and $m
}

rule Verifyd_Embedded_PE_Header {
    meta:
        description = "Embedded Windows PE header indicator"
        severity = "MEDIUM"
        category = "embedded_executable"
    strings:
        $mz = "MZ"
        $pe = "PE\x00\x00"
    condition:
        $mz at 0 or (#mz > 1 and #pe > 0)
}

rule Verifyd_Fake_PDF_Executable_Indicators {
    meta:
        description = "File contains both PDF and Windows executable indicators"
        severity = "HIGH"
        category = "masquerade"
    strings:
        $pdf = "%PDF-"
        $mz = "MZ"
        $pe = "PE\x00\x00"
    condition:
        $pdf at 0 and #mz > 0 and #pe > 0
}

rule Verifyd_Office_Macro_Indicator {
    meta:
        description = "Office macro project indicator"
        severity = "MEDIUM"
        category = "office"
    strings:
        $vba = "vbaProject.bin" nocase ascii wide
        $auto1 = "AutoOpen" nocase ascii wide
        $auto2 = "Document_Open" nocase ascii wide
    condition:
        $vba or any of ($auto*)
}

rule Verifyd_WScript_CScript_Download_Execution {
    meta:
        description = "Windows Script Host download or execution pattern"
        severity = "MEDIUM"
        category = "script"
    strings:
        $w1 = "wscript" nocase ascii wide
        $w2 = "cscript" nocase ascii wide
        $a = "CreateObject" nocase ascii wide
        $b = "WScript.Shell" nocase ascii wide
        $c = "MSXML2.XMLHTTP" nocase ascii wide
        $d = "ADODB.Stream" nocase ascii wide
    condition:
        any of ($w*) and $a and 1 of ($b,$c,$d)
}

rule Verifyd_Regsvr32_Scriptlet_Execution {
    meta:
        description = "regsvr32 scriptlet execution pattern"
        severity = "MEDIUM"
        category = "lolbin"
    strings:
        $r = "regsvr32" nocase ascii wide
        $s1 = "scrobj.dll" nocase ascii wide
        $s2 = "/i:" nocase ascii wide
        $s3 = "http://" nocase ascii wide
        $s4 = "https://" nocase ascii wide
    condition:
        $r and $s1 and $s2 and any of ($s3,$s4)
}

rule Verifyd_Suspicious_Batch_Downloader {
    meta:
        description = "Batch-file downloader or execution chain"
        severity = "MEDIUM"
        category = "script"
    strings:
        $b1 = "@echo off" nocase ascii wide
        $b2 = "cmd.exe" nocase ascii wide
        $d1 = "curl " nocase ascii wide
        $d2 = "bitsadmin" nocase ascii wide
        $d3 = "certutil" nocase ascii wide
        $x1 = "start " nocase ascii wide
        $x2 = "call " nocase ascii wide
    condition:
        any of ($b*) and any of ($d*) and any of ($x*)
}

rule Verifyd_Archive_Password_Indicator {
    meta:
        description = "Archive password lure indicator commonly used to hide malware from scanners"
        severity = "LOW"
        category = "archive"
    strings:
        $p1 = "password:" nocase ascii wide
        $p2 = "pass:" nocase ascii wide
        $p3 = "archive password" nocase ascii wide
        $p4 = "extract password" nocase ascii wide
        $a1 = ".zip" nocase ascii wide
        $a2 = ".rar" nocase ascii wide
        $a3 = ".7z" nocase ascii wide
    condition:
        any of ($p*) and any of ($a*)
}

rule Verifyd_Suspicious_LNK_Indicators {
    meta:
        description = "Windows shortcut indicator with suspicious command execution strings"
        severity = "MEDIUM"
        category = "shortcut"
    strings:
        $lnk = { 4C 00 00 00 01 14 02 00 }
        $p1 = "powershell" nocase ascii wide
        $p2 = "cmd.exe" nocase ascii wide
        $p3 = "wscript" nocase ascii wide
        $p4 = "mshta" nocase ascii wide
        $u1 = "http://" nocase ascii wide
        $u2 = "https://" nocase ascii wide
    condition:
        $lnk at 0 and any of ($p*) and any of ($u*)
}

rule Verifyd_Office_External_Template_Indicator {
    meta:
        description = "Office document external template or remote relationship indicator"
        severity = "MEDIUM"
        category = "office"
    strings:
        $r1 = "TargetMode=\"External\"" nocase ascii wide
        $r2 = "attachedTemplate" nocase ascii wide
        $u1 = "http://" nocase ascii wide
        $u2 = "https://" nocase ascii wide
    condition:
        any of ($r*) and any of ($u*)
}

rule Verifyd_HTML_Smuggling_Indicator {
    meta:
        description = "HTML smuggling-style JavaScript blob or download construction"
        severity = "MEDIUM"
        category = "html"
    strings:
        $h1 = "<html" nocase ascii wide
        $j1 = "atob(" nocase ascii wide
        $j2 = "Blob(" nocase ascii wide
        $j3 = "createObjectURL" nocase ascii wide
        $j4 = "download" nocase ascii wide
    condition:
        $h1 and 3 of ($j*)
}

rule Verifyd_Suspicious_Base64_PowerShell {
    meta:
        description = "PowerShell command containing long base64-like encoded content"
        severity = "MEDIUM"
        category = "script"
    strings:
        $ps = "powershell" nocase ascii wide
        $enc1 = "-enc" nocase ascii wide
        $enc2 = "EncodedCommand" nocase ascii wide
        $b64 = /[A-Za-z0-9+\/]{120,}={0,2}/ ascii
    condition:
        $ps and any of ($enc*) and $b64
}

rule Verifyd_Generic_Ransom_Note_Indicator {
    meta:
        description = "Ransom note language indicator"
        severity = "HIGH"
        category = "ransomware"
    strings:
        $r1 = "your files have been encrypted" nocase ascii wide
        $r2 = "all your files are encrypted" nocase ascii wide
        $r3 = "decrypt your files" nocase ascii wide
        $r4 = "bitcoin" nocase ascii wide
        $r5 = "private key" nocase ascii wide
    condition:
        2 of ($r*)
}

rule Verifyd_Suspicious_AutoRun_Indicator {
    meta:
        description = "Autorun persistence indicator"
        severity = "MEDIUM"
        category = "persistence"
    strings:
        $a1 = "[autorun]" nocase ascii wide
        $a2 = "open=" nocase ascii wide
        $a3 = "shellexecute=" nocase ascii wide
        $e1 = ".exe" nocase ascii wide
        $e2 = ".bat" nocase ascii wide
        $e3 = ".cmd" nocase ascii wide
        $e4 = ".vbs" nocase ascii wide
    condition:
        $a1 and any of ($a2,$a3) and any of ($e*)
}
