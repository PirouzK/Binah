import plotly.graph_objects as go
from ipywidgets import widgets, VBox, HBox, IntRangeSlider, SelectionSlider
import numpy as np

def interactive_map_analysis(scan, detector_name='sdd1'):
    """
    Creates an interactive dashboard for a MapScan using Plotly FigureWidgets.
    """
    # Load initial data
    spectra_2d = scan.get_sdd_data(detector_name)
    if spectra_2d is None:
        return None
    
    # Coordinates
    x = scan.x[:spectra_2d.shape[0]]
    y = scan.y[:spectra_2d.shape[0]]
    
    # 1. Create the Map Figure
    initial_roi = (80, 101)
    intensity = np.sum(spectra_2d[:, initial_roi[0]:initial_roi[1]], axis=1)
    
    map_fig = go.FigureWidget()
    map_trace = map_fig.add_scattergl(
        x=x, y=y, mode='markers',
        marker=dict(
            size=6,
            color=intensity,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Counts')
        ),
        name=detector_name
    )
    map_fig.update_layout(
        title=f"Map: {scan.scan_name} ({detector_name})",
        xaxis_title="Hexapod X",
        yaxis_title="Hexapod Y",
        dragmode='select',
        width=450, height=450,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    # 2. Create the Spectrum Figure
    initial_sum_spec = np.sum(spectra_2d, axis=0)
    bins = np.arange(spectra_2d.shape[1])
    
    spec_fig = go.FigureWidget()
    spec_trace = spec_fig.add_scatter(
        x=bins, y=initial_sum_spec, mode='lines',
        name='Summed Spectrum'
    )
    # Add vertical lines for ROI
    spec_fig.add_vline(x=initial_roi[0], line_dash="dash", line_color="red")
    spec_fig.add_vline(x=initial_roi[1], line_dash="dash", line_color="red")
    
    spec_fig.update_layout(
        title="Summed Spectrum (Selected Area)",
        xaxis_title="Bin Number",
        yaxis_title="Intensity",
        width=600, height=450
    )

    # --- Interaction Logic ---
    def on_selection(trace, points, selector):
        if not points.point_indices:
            new_spec = np.sum(spectra_2d, axis=0)
            spec_fig.layout.title.text = "Summed Spectrum (All Pixels)"
        else:
            indices = points.point_indices
            new_spec = np.sum(spectra_2d[indices, :], axis=0)
            spec_fig.layout.title.text = f"Summed Spectrum ({len(indices)} pixels selected)"
        
        with spec_fig.batch_update():
            spec_fig.data[0].y = new_spec

    map_fig.data[0].on_selection(on_selection)

    roi_slider = IntRangeSlider(
        value=initial_roi,
        min=0, max=spectra_2d.shape[1]-1,
        step=1,
        description='Spectral ROI:',
        style={'description_width': 'initial'},
        layout={'width': '100%'}
    )

    def on_roi_change(change):
        new_roi = change['new']
        new_intensity = np.sum(spectra_2d[:, new_roi[0]:new_roi[1]], axis=1)
        
        with map_fig.batch_update():
            map_fig.data[0].marker.color = new_intensity
            map_fig.layout.title.text = f"Map: {scan.scan_name} (ROI: {new_roi[0]}-{new_roi[1]})"
        
        with spec_fig.batch_update():
            if spec_fig.layout.shapes:
                spec_fig.layout.shapes[0].x0 = new_roi[0]
                spec_fig.layout.shapes[0].x1 = new_roi[0]
                spec_fig.layout.shapes[1].x0 = new_roi[1]
                spec_fig.layout.shapes[1].x1 = new_roi[1]

    roi_slider.observe(on_roi_change, names='value')

    det_dropdown = widgets.Dropdown(
        options=sorted(scan.sdd_files.keys()),
        value=detector_name,
        description='Detector:',
    )

    def on_det_change(change):
        nonlocal spectra_2d
        new_det = change['new']
        new_data = scan.get_sdd_data(new_det)
        if new_data is not None:
            spectra_2d = new_data
            on_roi_change({'new': roi_slider.value})
            on_selection(None, type('obj', (object,), {'point_indices': []}), None)

    det_dropdown.observe(on_det_change, names='value')

    controls = HBox([det_dropdown, roi_slider])
    dashboard = VBox([controls, HBox([map_fig, spec_fig])])
    return dashboard

def interactive_stack_analysis(stack, detector_names=None):
    """
    Creates an interactive dashboard for a StackScan with synchronized multi-detector selection.
    """
    if detector_names is None:
        detector_names = sorted(stack.sdd_files.keys())
    
    energies = sorted(stack.energies)
    if not energies:
        return None
    
    initial_energy = energies[0]
    initial_roi = (80, 101)
    current_selection = None
    
    # 1. Create Sliders (Shared)
    en_slider = SelectionSlider(
        options=[(f"{en:.2f} eV", en) for en in energies],
        value=initial_energy, description='Energy:', layout={'width': '45%'}
    )
    roi_slider = IntRangeSlider(
        value=initial_roi, min=0, max=255, step=1,
        description='Spectral ROI:', layout={'width': '45%'}
    )

    detector_plots = []
    
    # 2. Setup plots for each detector
    for det_name in detector_names:
        spectra_2d = stack.get_sdd_data(det_name, initial_energy)
        if spectra_2d is None:
            continue

        x = stack.x[:spectra_2d.shape[0]]
        y = stack.y[:spectra_2d.shape[0]]
        intensity = np.sum(spectra_2d[:, initial_roi[0]:initial_roi[1]], axis=1)
        
        # A. Map Figure
        map_fig = go.FigureWidget()
        map_fig.add_scattergl(
            x=x, y=y, mode='markers',
            marker=dict(size=5, color=intensity, colorscale='Viridis', showscale=True)
        )
        map_fig.update_layout(
            title=f"{det_name} Map @ {initial_energy:.2f} eV",
            xaxis_title="X", yaxis_title="Y",
            dragmode='select', width=350, height=350,
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        # B. PFY Figure
        def get_calc_pfy_func(dname):
            def calc_pfy(indices=None):
                pfy_vals = []
                current_roi = roi_slider.value
                for en in energies:
                    data = stack.get_sdd_data(dname, en)
                    if data is not None:
                        if indices is not None:
                            pfy_vals.append(np.sum(data[indices, current_roi[0]:current_roi[1]]))
                        else:
                            pfy_vals.append(np.sum(data[:, current_roi[0]:current_roi[1]]))
                    else:
                        pfy_vals.append(np.nan)
                return pfy_vals
            return calc_pfy

        calc_pfy_func = get_calc_pfy_func(det_name)
        initial_pfy = calc_pfy_func()
        
        spec_fig = go.FigureWidget()
        spec_fig.add_scatter(x=energies, y=initial_pfy, mode='lines+markers', name=det_name)
        spec_fig.add_vline(x=initial_energy, line_dash="dash", line_color="green")
        spec_fig.update_layout(
            title=f"{det_name} PFY",
            xaxis_title="Energy (eV)", yaxis_title="Counts",
            width=500, height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        detector_plots.append({
            'name': det_name,
            'map': map_fig,
            'spec': spec_fig,
            'calc_pfy': calc_pfy_func
        })

    # 3. Define Synchronized Selection Handler
    def update_all_pfys(indices):
        nonlocal current_selection
        current_selection = indices
        for item in detector_plots:
            new_pfy = item['calc_pfy'](indices)
            with item['spec'].batch_update():
                item['spec'].data[0].y = new_pfy
                item['spec'].layout.title.text = f"{item['name']} PFY (Selected Area)" if indices else f"{item['name']} PFY (All Pixels)"

    def sync_selection_handler(trace, points, selector):
        indices = points.point_indices if points.point_indices else None
        
        # Avoid recursive updates: only update if indices changed
        # We also want to highlight the selection on ALL maps for consistency
        with_selections = []
        for item in detector_plots:
            # We can't easily force selection on other ScatterGL traces programmatically 
            # without triggering more events, so we just update the data.
            pass
            
        update_all_pfys(indices)

    # Attach the same handler to ALL maps
    for item in detector_plots:
        item['map'].data[0].on_selection(sync_selection_handler)

    # 4. Define Global Observers
    def on_energy_change(change):
        new_en = change['new']
        for item in detector_plots:
            dname = item['name']
            data = stack.get_sdd_data(dname, new_en)
            if data is not None:
                roi = roi_slider.value
                new_intensity = np.sum(data[:, roi[0]:roi[1]], axis=1)
                with item['map'].batch_update():
                    item['map'].data[0].marker.color = new_intensity
                    item['map'].layout.title.text = f"{dname} @ {new_en:.2f} eV"
                with item['spec'].batch_update():
                    if item['spec'].layout.shapes:
                        item['spec'].layout.shapes[0].x0 = new_en
                        item['spec'].layout.shapes[0].x1 = new_en

    en_slider.observe(on_energy_change, names='value')

    def on_roi_change(change):
        current_en = en_slider.value
        for item in detector_plots:
            # Refresh map
            data = stack.get_sdd_data(item['name'], current_en)
            if data is not None:
                new_roi = change['new']
                new_intensity = np.sum(data[:, new_roi[0]:new_roi[1]], axis=1)
                with item['map'].batch_update():
                    item['map'].data[0].marker.color = new_intensity
            
            # Refresh PFY (respect current selection)
            new_pfy = item['calc_pfy'](current_selection)
            with item['spec'].batch_update():
                item['spec'].data[0].y = new_pfy

    roi_slider.observe(on_roi_change, names='value')

    # 5. Final Layout
    header = HBox([en_slider, roi_slider])
    rows = []
    for item in detector_plots:
        rows.append(HBox([item['map'], item['spec']]))
    
    dashboard = VBox([header] + rows)
    return dashboard
