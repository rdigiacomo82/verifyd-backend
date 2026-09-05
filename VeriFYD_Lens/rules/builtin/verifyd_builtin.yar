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
