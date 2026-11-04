"""Check public Horizons privacy text is visible in deployed website assets.

Best-effort. Absence from inspected bundles is UNVERIFIED, not proof
of absent disclosure. No credentials, accounts or uploads involved.
"""
import re
import ssl
from urllib.parse import urljoin
from urllib.request import Request, urlopen

HEADERS={"User-Agent":"VeriFYD-privacy-smoke/1.0"}
URLS=("https://vfvid.com/voice", "https://vfvid.com/privacy")
PATTERNS=(
    "Voicemail recordings and optional analysis",
    "external speech-to-text provider",
    "currently OpenAI",
    "Analyze &amp; Transcribe",
)
found=[]
for url in URLS:
    try:
        with urlopen(Request(url,headers=HEADERS),timeout=25) as req:
            html=req.read(2_000_000).decode("utf-8","replace")
            print("Page:",url,req.status,"html bytes",len(html))
        paths=re.findall(r'(?:src|href)=["\']([^"\']+\.js(?:\?[^"\']*)?)["\']',html)
        pages=[html]
        print("top-level JS refs:",len(paths))
        for path in dict.fromkeys(paths[:40]):
            try:
                link=urljoin(url,path)
                with urlopen(Request(link,headers=HEADERS),timeout=20) as req:
                    pages.append(req.read(2_500_000).decode("utf-8","replace"))
            except Exception as error:
                print("Bundle unreadable:",path,type(error).__name__)
        joined=" ".join(pages).lower()
        flags=[x for x in PATTERNS if x.lower() in joined]
        print("Found disclosure indicators:",flags)
        if len(flags)>=2:
            found.append(url)
    except Exception as e:
        print("Page inaccessible:",url,type(e).__name__,str(e)[:160])
if found:
    print("PASS: Public content includes voicemail disclosure indicators on:",found)
else:
    print("UNVERIFIED: Public scripts scanned did not establish deployed voicemail disclosure.")
    print("Do not activate private-audio transcription merely based on this result.")
