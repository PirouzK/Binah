# orca_viewer.spec  —  PyInstaller build spec for ORCA TDDFT Viewer
# Build with:  pyinstaller orca_viewer.spec

import sys
block_cipher = None

a = Analysis(
    ['main_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include supporting_information.pdf if you want it bundled
        # ('supporting_information.pdf', '.'),
    ],
    hiddenimports=[
        # matplotlib backends
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.backends._backend_tk',
        # scipy submodules
        'scipy.special',
        'scipy.special._ufuncs',
        'scipy.interpolate',
        'scipy.optimize',
        'scipy.signal',
        # larch / xraylarch
        'larch',
        'larch.xafs',
        'larch.xafs.pre_edge',
        'larch.math',
        'larch.fitting',
        # h5py (Athena .prj support)
        'h5py',
        'h5py._conv',
        # other
        'seaborn',
        'PIL',
        'PIL.Image',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.colorchooser',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Not needed — keeps the bundle smaller
        'IPython',
        'ipykernel',
        'jupyter',
        'notebook',
        'pandas',
        'pymatgen',
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
    name='ORCA_TDDFT_Viewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,      # No terminal window for GUI app
    windowed=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Uncomment and provide icon.ico / icon.icns
)

# macOS: wrap in a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='ORCA_TDDFT_Viewer.app',
        icon=None,          # Replace with 'icon.icns' if you have one
        bundle_identifier='ca.ucalgary.orca-tddft-viewer',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleDisplayName': 'ORCA TDDFT Viewer',
            'NSHighResolutionCapable': True,
        },
    )
