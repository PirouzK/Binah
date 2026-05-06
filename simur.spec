# simur.spec - PyInstaller build spec for Simur
# Build with:  python -m PyInstaller simur.spec

import sys

block_cipher = None

a = Analysis(
    ['simur.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('Simur.png', '.'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends._backend_tk',
        'mpl_toolkits.mplot3d',
        'mpl_toolkits.mplot3d.art3d',
        'skimage.measure',
        'cclib',
        'cclib.method.volume',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Simur',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    windowed=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icons/Simur.ico' if sys.platform == 'win32' else 'assets/icons/Simur.icns',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='Simur.app',
        icon='assets/icons/Simur.icns',
        bundle_identifier='ca.ucalgary.simur',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleDisplayName': 'Simur',
            'NSHighResolutionCapable': True,
        },
    )
