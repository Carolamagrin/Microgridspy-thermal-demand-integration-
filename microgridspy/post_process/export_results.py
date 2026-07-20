import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict

from microgridspy.model.model import Model
from microgridspy.post_process.data_retrieval import get_sizing_results

def _n_scenarios(da) -> int:
    return da.sizes.get("scenarios", 1)

def _isel_scenario(da, scenario: int):
    return da.isel(scenarios=scenario) if "scenarios" in getattr(da, "dims", ()) else da

def save_energy_balance_to_excel(model: Model, base_filepath: Path) -> None:
    demand = model.parameters['DEMAND']
    thermal_demand = model.parameters.get('THERMAL_DEMAND', None)
    res_production = model.get_solution_variable('Energy Production by Renewables')
    curtailment = model.get_solution_variable('Curtailment by Renewables')
    res_conversion_losses = model.get_solution_variable('Conversion Losses - Renewable Sources')    

    res_to_ac_bus = model.get_solution_variable('Renewable Energy to AC Bus')
    res_to_thermal_dc_bus = model.get_solution_variable('Renewable Energy to Thermal DC Bus')

    # Mapping steps to years
    steps = res_production.coords['steps'].values
    years = demand.coords['years'].values
    start_year = years[0]
    step_duration = model.settings.advanced_settings.step_duration
    years_steps_tuples = [(years[i], steps[i // step_duration]) for i in range(len(years))]

    for scenario in range(_n_scenarios(demand)):
        filepath = base_filepath / f"Energy Balance - Scenario {scenario + 1}.xlsx"
        with pd.ExcelWriter(filepath) as writer:
            # Write energy balance for each year
            for year in range(len(years)):
                step = years_steps_tuples[year][1]
                demand_y = _isel_scenario(demand, scenario).isel(years=year).values
                data = {'Electric Demand (kWh)': demand_y / 1000}

                if thermal_demand is not None:
                    thermal_demand_y = _isel_scenario(thermal_demand, scenario).isel(years=year).values
                    data['Thermal Demand (kWh_th)'] = thermal_demand_y / 1000
                
                # Add specific production for each renewable source
                for source in res_production.coords['renewable_sources'].values:
                    source_production = res_production.isel(scenarios=scenario).sel(steps=step, renewable_sources=source).values / 1000
                    source_curtailment = curtailment.isel(scenarios=scenario).sel(years=year + start_year, renewable_sources=source).values / 1000 if curtailment is not None else 0
                    res_losses = res_conversion_losses.isel(scenarios=scenario).sel(years=year + start_year, renewable_sources=source).values / 1000
                    res_losses = res_losses.flatten() 
                    source_to_ac = res_to_ac_bus.isel(scenarios=scenario).sel(years=year + start_year, renewable_sources=source).values / 1000
                    source_to_dc_thermal = res_to_thermal_dc_bus.isel(scenarios=scenario).sel(years=year + start_year, renewable_sources=source).values / 1000
                    data[f'{source} Total Production (kWh)'] = source_production
                    data[f'{source} Curtailment (kWh)'] = source_curtailment
                    data[f'{source} Actual Production (kWh)'] = source_production - source_curtailment
                    data[f'{source} Conversion Losses (kWh)'] = res_losses
                    data[f'{source} To AC Bus (kWh)'] = source_to_ac
                    data[f'{source} To Thermal DC Bus (kWh)'] = source_to_dc_thermal


                # Battery data
                if model.has_battery:
                    battery_inflow = model.get_solution_variable('Battery Inflow')
                    battery_outflow = model.get_solution_variable('Battery Outflow')
                    state_of_charge = model.get_solution_variable('Battery State of Charge')
                    battery_units = model.get_solution_variable('Unit of Nominal Capacity for Batteries')
                    battery_nominal_capacity = model.parameters['BATTERY_NOMINAL_CAPACITY']

                    data['Battery Outflow (kWh)'] = (battery_outflow.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                    data['Battery Inflow (kWh)'] = (battery_inflow.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                    data['Battery State of Charge (%)'] = (
                        state_of_charge.isel(scenarios=scenario)
                        .sel(years=year + start_year).values
                        / (battery_units.sel(steps=step).values * battery_nominal_capacity.values)
                        * 100
                    )

                    battery_losses = model.get_solution_variable("Conversion Losses - Battery")
                    battery_losses = (
                        battery_losses.isel(scenarios=scenario)
                        .sel(years=year + start_year).values
                    ) / 1000
                    battery_losses = battery_losses.flatten()
                    data['Battery Conversion Losses (kWh)'] = battery_losses
                    # Thermal DC battery data
                    thermal_battery_inflow = model.get_solution_variable('Thermal DC Battery Inflow')
                    thermal_battery_outflow = model.get_solution_variable('Thermal DC Battery Outflow')
                    thermal_battery_soc = model.get_solution_variable('Thermal DC Battery State of Charge')
                    thermal_battery_units = model.get_solution_variable('Unit of Nominal Capacity for Thermal DC Battery')

                    thermal_battery_capacity = (
                        thermal_battery_units.sel(steps=step).values
                        * battery_nominal_capacity.values
                    )

                    data['Thermal Battery Outflow (kWh)'] = (
                        thermal_battery_outflow.isel(scenarios=scenario)
                        .sel(years=year + start_year).values
                    ) / 1000

                    data['Thermal Battery Inflow (kWh)'] = (
                        thermal_battery_inflow.isel(scenarios=scenario)
                        .sel(years=year + start_year).values
                    ) / 1000

                    if thermal_battery_capacity > 0:
                        data['Thermal Battery State of Charge (%)'] = (
                            thermal_battery_soc.isel(scenarios=scenario)
                            .sel(years=year + start_year).values
                            / thermal_battery_capacity
                            * 100
                        )
                    else:
                        data['Thermal Battery State of Charge (%)'] = 0.0

                    data['Thermal Battery Installed Capacity (kWh)'] = thermal_battery_capacity / 1000

                # TES data
                if model.has_tes:

                    tes_charge = model.get_solution_variable("TES Charge Flow")
                    tes_discharge = model.get_solution_variable("TES Discharge Flow")
                    tes_soc = model.get_solution_variable("TES State of Charge")
                    tes_ice_production = model.get_solution_variable("TES Ice Production")
                    tes_electric = model.get_solution_variable("TES Electric Consumption")

                    # Q per kg: prova prima il nome che hai nei parametri; fallback se cambia naming
                    if "TES_Q_PER_KG" in model.parameters:
                        tes_q_per_kg = model.parameters["TES_Q_PER_KG"].values
                    elif "TES_Q_PER_KG_TS" in model.parameters:
                        tes_q_per_kg = model.parameters["TES_Q_PER_KG_TS"].sel(years=year + start_year).values
                    else:
                        raise KeyError("TES_Q_PER_KG (o TES_Q_PER_KG_TS) non trovato in model.parameters")

                    tes_soc_y = _isel_scenario(tes_soc, scenario).sel(years=year + start_year).values
                    tes_charge_y = _isel_scenario(tes_charge, scenario).sel(years=year + start_year).values
                    tes_discharge_y = _isel_scenario(tes_discharge, scenario).sel(years=year + start_year).values
                    tes_ice_y = _isel_scenario(tes_ice_production, scenario).sel(years=year + start_year).values
                    tes_el_y = _isel_scenario(tes_electric, scenario).sel(years=year + start_year).values

                    data["TES State of Charge (kg)"] = tes_soc_y

                    # Capacità TES: può essere un parametro o una variabile di sizing.
                    # 1) se è parametro (fisso)
                    tes_capacity_value = None
                    # TES compressor capacity
                    tes_compressor_capacity_value = None

                    # 1) prova come variabile di soluzione
                    for var_name in [
                        "TES Compressor Capacity",
                        "TES_Compressor_Capacity",
                        "Ice Maker Capacity",
                        "Compressor Capacity for TES"
                    ]:
                        try:
                            cap_sol = model.get_solution_variable(var_name)
                            tes_compressor_capacity_value = float(_isel_scenario(cap_sol, scenario).values)
                            break
                        except Exception:
                            pass

                    # 2) fallback come parametro
                    if tes_compressor_capacity_value is None:
                        for par_name in [
                            "TES_COMPRESSOR_CAPACITY",
                            "ICE_MAKER_CAPACITY",
                            "TES_COMPRESSOR_CAPACITY_MAX"
                        ]:
                            if par_name in model.parameters:
                                cap_da = model.parameters[par_name]
                                tes_compressor_capacity_value = float(_isel_scenario(cap_da, scenario).values)
                                break

                    if "TES_CAPACITY" in model.parameters:
                        # può essere scalare o indicizzato
                        cap_da = model.parameters["TES_CAPACITY"]
                        tes_capacity_value = float(_isel_scenario(cap_da, scenario).values)
                    elif "TES_CAPACITY_MAX" in model.parameters:
                        cap_da = model.parameters["TES_CAPACITY_MAX"]
                        tes_capacity_value = float(_isel_scenario(cap_da, scenario).values)
                    else:
                        # 2) se è variabile (ottimizzata): prova a leggerla dalla solution
                        # (questo nome è quello tipico che hai nel modello linopy)
                        try:
                            cap_sol = model.get_solution_variable("TES Capacity")
                            tes_capacity_value = float(_isel_scenario(cap_sol, scenario).values)
                        except Exception:
                            # 3) fallback: usa il massimo SOC come "capacità effettiva" per non bloccare l’export
                            tes_capacity_value = float(pd.Series(tes_soc_y).max())

                    # percentuale SOC solo se capacità > 0
                    if tes_capacity_value and tes_capacity_value > 0:
                        data["TES Capacity (kg)"] = tes_capacity_value
                        data["TES State of Charge (%)"] = tes_soc_y / tes_capacity_value * 100
                    else:
                        data["TES Capacity (kg)"] = tes_capacity_value
                        data["TES State of Charge (%)"] = 0.0

                    data["TES Charge (kg/h)"] = tes_charge_y
                    data["TES Discharge (kg/h)"] = tes_discharge_y
                    data["TES Ice Production (kg/h)"] = tes_ice_y
                    data["TES Compressor Capacity (kW)"] = tes_compressor_capacity_value/1000

                    # elettrico: stai dividendo per 1000 -> coerente con il resto del file (Wh -> kWh)
                    data["TES Electric Consumption (kWh)"] = tes_el_y / 1000

                    # Cooling output:
                    # se tes_discharge è kg/h e Q_per_kg è Wh/kg -> W. Per ottenere kWh su base oraria: /1000.
                    data["TES Cooling Output (kWh_th)"] = tes_discharge_y * tes_q_per_kg / 1000
                    
                # Direct compressor data
                if model.has_compressor:

                    direct_electric = model.get_solution_variable("Compressor Electric Consumption")
                    direct_cooling = model.get_solution_variable("Compressor Cooling Output")
                    direct_capacity = model.get_solution_variable("Compressor Capacity")

                    # Capacità nominale del compressore diretto
                    data["Direct Compressor Capacity (W)"] = (
                        direct_capacity.values
                    )

                    data["Direct Compressor Electric Consumption (kWh)"] = (
                        _isel_scenario(direct_electric, scenario)
                        .sel(years=year + start_year).values / 1000
                    )

                    data["Direct Cooling Output (kWh_th)"] = (
                        _isel_scenario(direct_cooling, scenario)
                        .sel(years=year + start_year).values / 1000
                    )
            
                # Generator data
                if model.has_generator:
                    generator_production = model.get_solution_variable('Generator Energy Production')
                    generator_conversion_losses = model.get_solution_variable('Conversion Losses - Generator')
                    if model.settings.generator_params.partial_load == True:
                        fuel_consumption = model.get_solution_variable('Generator Fuel Consumption')
                    else:
                        fuel_consumption = generator_production / (model.parameters['GENERATOR_NOMINAL_EFFICIENCY'] * model.parameters['FUEL_LHV'])
                    for gen_type in generator_production.coords['generator_types'].values:
                        data[f'{gen_type} Production (kWh)'] = (generator_production.isel(scenarios=scenario).sel(years=year + start_year, generator_types=gen_type).values) / 1000
                        data[f'{gen_type} Fuel Consumption (liter)'] = (fuel_consumption.isel(scenarios=scenario).sel(years=year + start_year, generator_types=gen_type).values)
                        generator_losses = (generator_conversion_losses.isel(scenarios=scenario).sel(years=year + start_year, generator_types=gen_type).values) / 1000
                        generator_losses = generator_losses.flatten() 
                        data[f'{gen_type} Conversion Losses (kWh)'] = generator_losses

                
                # Grid connection data
                if model.has_grid_connection:
                    energy_from_grid = model.get_solution_variable('Energy from Grid')
                    data['Energy from Grid (kWh)'] = (energy_from_grid.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                    if model.settings.advanced_settings.grid_connection_type == 1:
                        energy_to_grid = model.get_solution_variable('Energy to Grid')
                        data['Energy to Grid (kWh)'] = (energy_to_grid.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                    grid_conversion_losses = model.get_solution_variable('Conversion Losses - Grid')
                    grid_losses = (grid_conversion_losses.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                    grid_losses = grid_losses.flatten() 
                    data['Grid Conversion Losses (kWh)'] = grid_losses

                # Lost load data
                if model.get_settings('lost_load_fraction') > 0.0:
                    lost_load = model.get_solution_variable('Lost Load')
                    data['Lost Load (kWh)'] = (lost_load.isel(scenarios=scenario).sel(years=year + start_year).values) / 1000
                
                try:
                    thermal_unmet = model.get_solution_variable('Thermal Unmet Demand')
                    data['Thermal Unmet Demand (kWh_th)'] = (
                    _isel_scenario(thermal_unmet, scenario)
                    .sel(years=year + start_year).values / 1000
                )
                except ValueError:
                    data['Thermal Unmet Demand (kWh_th)'] = [0.0] * len(demand_y)
                df = pd.DataFrame(data)
                df = df.round(2)  # Round all numerical values to 2 decimal places
                df.to_excel(writer, sheet_name=f'Year {year + 1}', index=False)


def save_plots(plots_filepath: Path, figures: Dict[str, plt.Figure]):
    """
    Save all plots generated in the dashboard to separate files.

    Args:
    model (Model): The model object containing all the data.
    plots_filepath (Path): The directory path where plots should be saved.
    figures (Dict[str, plt.Figure]): A dictionary containing all the generated figures.
    """

    for plot_name, fig in figures.items():
        # Clean the plot name to use as a filename
        filename = "".join(x for x in plot_name if x.isalnum() or x in [' ', '_']).rstrip()
        filename = filename.replace(' ', '_') + '.png'
        
        # Save the figure
        fig.savefig(plots_filepath / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory