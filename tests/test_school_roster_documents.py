"""Contract tests for Principal-only roster document preview."""
import io
import inspect
import unittest
from docx import Document
from reportlab.pdfgen import canvas
import trust_voice_school_roster_documents as roster_documents


class SchoolRosterDocumentParserTests(unittest.TestCase):
    def test_word_table_maps_school_student_columns(self):
        document=Document()
        table=document.add_table(rows=1, cols=5)
        for cell,value in zip(table.rows[0].cells,
            ["Student ID","First Name","Last Name","Primary Guardian Email","Backup Guardian Email"]):
            cell.text=value
        cells=table.add_row().cells
        for cell,value in zip(cells,
            ["DEMO1001","Taylor","Example","primary@example.test","backup@example.test"]):
            cell.text=value
        output=io.BytesIO()
        document.save(output)
        rows=roster_documents._docx_rows(output.getvalue())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["local_ref"],"DEMO1001")
        self.assertEqual(rows[0]["first_name"],"Taylor")
        self.assertEqual(rows[0]["last_name"],"Example")
        self.assertEqual(rows[0]["primary_guardian_email"],"primary@example.test")
        self.assertEqual(rows[0]["backup_guardian_email"],"backup@example.test")

    def test_scanned_pdf_is_not_interpreted_as_student_record(self):
        output=io.BytesIO()
        doc=canvas.Canvas(output,pagesize=(612,792))
        doc.drawString(70,700,"A non-tabular school letter is not a roster")
        doc.showPage()
        doc.save()
        self.assertEqual(roster_documents._pdf_rows(output.getvalue()),[])

    def test_pipe_separated_text_pdf_with_headers(self):
        output=io.BytesIO()
        doc=canvas.Canvas(output,pagesize=(612,792))
        doc.setFont("Courier",10)
        doc.drawString(40,750,"local_ref | first_name | last_name")
        doc.drawString(40,730,"DEMO101 | Jordan | Example")
        doc.showPage()
        doc.save()
        rows=roster_documents._pdf_rows(output.getvalue())
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["local_ref"],"DEMO101")
        self.assertEqual(rows[0]["first_name"],"Jordan")
        self.assertEqual(rows[0]["last_name"],"Example")

    def test_rejects_duplicate_ambiguous_headers_and_invalid_ids(self):
        with self.assertRaises(ValueError):
            roster_documents._map_header(["student id","local_ref","first name","last name"])
        with self.assertRaises(ValueError):
            roster_documents._collect_rows([
                ["local_ref","first_name","last_name"],
                ["student 001","James","Example"]])

    def test_endpoint_restricts_school_access_and_is_preview_only(self):
        src=inspect.getsource(roster_documents.parse_school_roster_document)
        self.assertIn("_require_enrollment_enabled",src)
        self.assertIn("_school_admin",src)
        self.assertIn("await file.read(_MAX_BYTES+1)",src)
        self.assertIn('"preview_only":True',src)
        self.assertIn('"stored":False',src)
        self.assertIn('"messages_sent":False',src)
        self.assertNotIn("INSERT INTO",src)
        self.assertNotIn("send_invitation",src)


if __name__=="__main__":
    unittest.main()
