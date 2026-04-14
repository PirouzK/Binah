from .scans import MapScan, StackScan
from .plotting import plot_xeol
from .interactive import interactive_map_analysis, interactive_stack_analysis


def launch_gui():
    from .gui import main

    main()
