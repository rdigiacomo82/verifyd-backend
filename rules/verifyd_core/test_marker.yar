rule VeriFYD_Phase2A_Test_Marker : verifyd test {
    meta:
        verifyd_schema = "1"
        description = "Harmless marker used to validate the VeriFYD YARA-X pipeline"
        severity = "informational"
        category = "engine_test"
        confidence = 100
        author = "VeriFYD"
        source = "verifyd_core"
        tags = "verifyd,test"
        dedupe_key = "verifyd_phase2a_test_marker"
    strings:
        $marker = "VERIFYD_YARAX_TEST_MARKER_2026" ascii
    condition:
        $marker
}
