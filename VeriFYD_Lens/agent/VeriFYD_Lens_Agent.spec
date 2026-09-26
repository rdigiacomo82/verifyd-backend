# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
_yx_datas, _yx_binaries, _yx_hiddenimports = collect_all("yara_x")


a = Analysis(
    ['lens_agent.py'],
    pathex=['../..'],
    binaries=_yx_binaries,
    datas=[('../../rules', 'rules')] + _yx_datas,
    hiddenimports=_yx_hiddenimports + ["yara_x"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VeriFYD_Lens_Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
