import subprocess

def stamp_video(input_path, output_path, cert_id):

    vf = (
        f"drawtext=text='VeriFYD':x=10:y=10:fontsize=24:"
        f"fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4,"
        f"drawtext=text='ID:{cert_id}':x=w-tw-20:y=h-th-20:fontsize=16:"
        f"fontcolor=white@0.85:box=1:boxcolor=black@0.4:boxborderw=4"
    )

    cmd = [
        "ffmpeg","-y",
        "-i", input_path,
        "-vf", vf,
        "-map","0:v:0",
        "-map","0:a?",
        "-c:v","libx264",
        "-preset","fast",
        "-crf","23",
        "-c:a","copy",
        "-movflags","+faststart",
        output_path
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[-300:])
