import json
import tkinter as tk
from tkinter import messagebox
from urllib import request, error

AGENT = "http://127.0.0.1:8765"

def req(path, payload=None, timeout=15):
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    r = request.Request(AGENT + path, data=data, headers=headers, method=method)
    with request.urlopen(r, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def activate():
    email = email_var.get().strip()
    token = token_var.get().strip()
    if not email or "@" not in email:
        messagebox.showerror("VeriFYD Lens", "Enter the purchaser email address.")
        return
    if not token:
        messagebox.showerror("VeriFYD Lens", "Enter the Lens activation token.")
        return
    try:
        status, body = req("/activation/activate", {"email": email, "entitlement_token": token}, 30)
        if status == 200 and body.get("activated"):
            messagebox.showinfo("VeriFYD Lens", "Activation complete. VeriFYD Lens is ready to use.")
            root.destroy()
            return
    except error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
            detail = body.get("detail") or "Activation could not be verified."
        except Exception:
            detail = "Activation could not be verified."
        messagebox.showerror("VeriFYD Lens", detail)
        return
    except Exception:
        messagebox.showerror("VeriFYD Lens", "The VeriFYD Lens Agent is not running.")
        return
    messagebox.showerror("VeriFYD Lens", "Activation could not be verified.")

root = tk.Tk()
root.title("Activate VeriFYD Lens")
root.geometry("520x275")
root.resizable(False, False)
tk.Label(root, text="VeriFYD Lens Activation", font=("Segoe UI", 17, "bold")).pack(pady=(20, 4))
tk.Label(root, text="Enter the email used for purchase and the activation token provided after checkout.",
         wraplength=460, justify="center", font=("Segoe UI", 10)).pack(pady=(0, 14))
form = tk.Frame(root)
form.pack(fill="x", padx=34)
tk.Label(form, text="Purchaser email", anchor="w").pack(fill="x")
email_var = tk.StringVar()
tk.Entry(form, textvariable=email_var, font=("Segoe UI", 10)).pack(fill="x", pady=(2, 10))
tk.Label(form, text="Activation token", anchor="w").pack(fill="x")
token_var = tk.StringVar()
tk.Entry(form, textvariable=token_var, font=("Consolas", 9), show="*").pack(fill="x", pady=(2, 16))
tk.Button(root, text="Activate VeriFYD Lens", command=activate, width=24).pack()
root.mainloop()
