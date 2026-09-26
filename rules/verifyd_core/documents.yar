rule VeriFYD_PDF_Embedded_JavaScript_Indicators : verifyd pdf javascript {
    meta:
        verifyd_schema = "1"
        description = "Detects PDF objects containing common embedded JavaScript action indicators"
        severity = "medium"
        category = "document_active_content"
        confidence = 70
        author = "VeriFYD"
        source = "verifyd_core"
        tags = "pdf,javascript,active_content"
        dedupe_key = "pdf_embedded_javascript"
    strings:
        $pdf = "%PDF-" ascii
        $js1 = "/JavaScript" ascii
        $js2 = "/JS" ascii
    condition:
        $pdf at 0 and 1 of ($js*)
}
