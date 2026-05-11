# binah.spec  —  PyInstaller build spec for Binah
# Build with:  pyinstaller binah.spec

import sys
from PyInstaller.utils.hooks import collect_data_files
block_cipher = None

a = Analysis(
    ['binah.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('Binah.png', '.'),
        *collect_data_files('tkinterdnd2'),
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
        'tkinterdnd2',
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
    name='Binah',
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
    icon='assets/icons/Binah.ico' if sys.platform == 'win32' else 'assets/icons/Binah.icns',
)

# macOS: wrap in a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='Binah.app',
        icon='assets/icons/Binah.icns',
        bundle_identifier='ca.ucalgary.binah',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleDisplayName': 'Binah',
            'NSHighResolutionCapable': True,
        },
    )
