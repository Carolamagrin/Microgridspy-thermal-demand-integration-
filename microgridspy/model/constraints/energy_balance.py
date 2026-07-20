from typing import Dict

import xarray as xr
import linopy
from linopy import Model

from microgridspy.model.parameters import ProjectParameters


def add_energy_balance_constraints(
    model: Model, 
    settings: ProjectParameters, 
    sets: xr.Dataset, 
    param: xr.Dataset, 
    var: Dict[str, linopy.Variable],
    has_battery: bool,
    has_generator: bool,
    has_compressor: bool,
    has_grid_connection: bool,
    has_tes: bool) -> None:
    """Add energy balance constraint."""
    years = sets.years.values
    steps = sets.steps.values
    step_duration = settings.advanced_settings.step_duration
    milp_formulation = settings.advanced_settings.milp_formulation
    years_steps_tuples = [(years[i] - years[0], steps[i // step_duration]) for i in range(len(years))]
    # Calculate total renewable energy production
    
    for year in sets.years.values:
        step = years_steps_tuples[year - years[0]][1]
        
        # AC bus production: only renewable energy sent to the AC bus
        yearly_energy_production: linopy.LinearExpression = var['res_to_ac_bus'].sum('renewable_sources').sel(years=year)

        yearly_conversion_losses: linopy.LinearExpression = 0

        res_yearly_conversion_losses = {}

        for res in sets.renewable_sources.values:

            # Split renewable production between AC bus, thermal DC bus and curtailment
            model.add_constraints(
                var['res_energy_production'].sel(steps=step, renewable_sources=res)
                ==
                var['res_to_ac_bus'].sel(years=year, renewable_sources=res)
                + var['res_to_thermal_dc_bus'].sel(years=year, renewable_sources=res)
                + var['curtailment'].sel(years=year, renewable_sources=res),
                name=f"RES Energy Split - Year {year} - {res}"
            )

            # Inverter losses are applied only to the energy sent to the AC bus
            source_losses = (
                var['res_to_ac_bus'].sel(renewable_sources=res, years=year)
                * (1 - param['RES_INVERTER_EFFICIENCY'].sel(renewable_sources=res))
            )

            res_yearly_conversion_losses[res] = source_losses

            model.add_constraints(
                source_losses == var['res_conversion_losses'].sel(renewable_sources=res, years=year),
                name=f"RES ConversionLosses - {res} - Year {year}"
            )
        yearly_conversion_losses += sum(res_yearly_conversion_losses.values())

        if has_battery:

            # AC battery contribution to the AC bus
            battery_system_energy = (
                var['battery_outflow'].sel(years=year)
                - var['battery_inflow'].sel(years=year)
            )

            yearly_energy_production += battery_system_energy

            # Inverter losses associated only with the AC battery
            battery_losses = (
                var['battery_outflow'].sel(years=year)
                * (1 - param['BATTERY_INVERTER_EFFICIENCY_DC_AC'].item())
                +
                var['battery_inflow'].sel(years=year)
                * ((1 / param['BATTERY_INVERTER_EFFICIENCY_AC_DC'].item()) - 1)
            )

            model.add_constraints(
                battery_losses == var['battery_conversion_losses'].sel(years=year),
                name=f"Battery ConversionLosses - Year {year}"
            )

            yearly_conversion_losses += battery_losses
                

        if has_generator:
            for generator in sets.generator_types.values:
                yearly_energy_production += var['generator_energy_production'].sel(years=year, generator_types=generator)
                generator_loss = (
                    var['generator_energy_production'].sel(generator_types=generator, years=year)
                    * (1 - param['GENERATOR_RECTIFIER_EFFICIENCY'].sel(generator_types=generator))
                )
                
                # Add constraint for each generator type's conversion losses
                model.add_constraints(
                    generator_loss == var['generator_conversion_losses'].sel(generator_types=generator, years=year),
                    name=f"Generator ConversionLosses - {generator} - Year {year}"
                )
                yearly_conversion_losses += generator_loss

        if has_grid_connection:
            if settings.advanced_settings.grid_connection_type == 1:
                # Calculate energy from grid and energy to grid if Purchase/Sell is selected
                yearly_energy_production += (var['energy_from_grid'].sel(years=year) - var['energy_to_grid'].sel(years=year))
                grid_losses = (
                    var['energy_from_grid'].sel(years=year) * (1 - param['GRID_TO_MICROGRID_EFFICIENCY'])
                    + var['energy_to_grid'].sel(years=year) * ((1 / param['MICROGRID_TO_GRID_EFFICIENCY']) - 1)
                )
            else:
                # Calculate energy from grid if Purchase Only is selected
                yearly_energy_production += var['energy_from_grid'].sel(years=year)
                grid_losses = var['energy_from_grid'].sel(years=year) * (1 - param['GRID_TO_MICROGRID_EFFICIENCY'])
            model.add_constraints(grid_losses == var['grid_conversion_losses'].sel(years=year), name=f"Grid ConversionLosses - Year {year}")

            yearly_conversion_losses += grid_losses

        if settings.project_settings.lost_load_fraction > 0:
            yearly_energy_production += var['lost_load'].sel(years=year)

        # Domanda elettrica totale: domanda base + compressore diretto + compressore TES
        base_electric_demand = param['DEMAND'].sel(years=year)

         
        # Energy balance per ogni anno:
        # AC energy balance: only village electric demand is supplied by the AC bus
        model.add_constraints(
            yearly_energy_production - yearly_conversion_losses == base_electric_demand,
            name=f"AC Energy Balance Constraint - Year {year}"
        )

        # DC thermal electrical demand: compressor + TES compressor + TES pump
        if has_compressor or has_tes:

            if has_tes:
                tes_pump_electric_demand = (
                    param['TES_PUMP_SPECIFIC_CONSUMPTION']
                    * var['tes_discharge'].sel(years=year)
                    * param['TES_Q_PER_KG']
                )
            else:
                tes_pump_electric_demand = 0 * yearly_energy_production

            if has_compressor and has_tes:
                dc_thermal_electric_demand = (
                    var['compressor_electric_consumption'].sel(years=year)
                    + var['tes_electric_consumption'].sel(years=year)
                    + tes_pump_electric_demand
                )
            elif has_compressor:
                dc_thermal_electric_demand = var['compressor_electric_consumption'].sel(years=year)
            elif has_tes:
                dc_thermal_electric_demand = (
                    var['tes_electric_consumption'].sel(years=year)
                    + tes_pump_electric_demand
                )

            dc_thermal_supply = (
                var['res_to_thermal_dc_bus'].sum('renewable_sources').sel(years=year)
                + var['thermal_battery_outflow'].sel(years=year)
                - var['thermal_battery_inflow'].sel(years=year)
            )

            model.add_constraints(
                dc_thermal_supply == dc_thermal_electric_demand,
                name=f"DC Thermal Electrical Balance - Year {year}"
            )

        # Bilancio della domanda di freddo (se c'è compressore e/o TES)
        if has_compressor or has_tes:
            # Freddo diretto dal compressore (se presente)
            if has_compressor:
                cooling_direct = var['compressor_cooling_output'].sel(years=year)
            else:
                cooling_direct = 0

            # Freddo dal TES: m_discharge * Q_per_kg (se TES presente)
            if has_tes:
                cooling_from_tes = var['tes_discharge'].sel(years=year) * param['TES_Q_PER_KG']
            else:
                cooling_from_tes = 0


            # Domanda termica da soddisfare
            thermal_demand = param['THERMAL_DEMAND'].sel(years=year)

            thermal_unmet_var = var.get('thermal_unmet_demand', None)

            if thermal_unmet_var is not None:
                thermal_unmet = thermal_unmet_var.sel(years=year)
            else:
                thermal_unmet = 0

            model.add_constraints(
                cooling_direct + cooling_from_tes + thermal_unmet == thermal_demand,
                name=f"Cooling Demand Balance - Year {year}"
            )

            if thermal_unmet_var is not None:
                model.add_constraints(
                    thermal_unmet <= param['LOST_LOAD_FRACTION'] * thermal_demand,
                    name=f"Thermal Lost Load Constraint - Year {year}"
            )

    # Add renewable penetration constraint if specified
    if settings.project_settings.renewable_penetration > 0:
        add_renewable_penetration_constraint(model, settings, sets, param, var, has_battery, has_generator, has_grid_connection)

    if settings.project_settings.lost_load_fraction > 0:
        add_lost_load_constraint(
            model, settings, sets, param, var,
            has_battery, has_generator, has_compressor, has_grid_connection, has_tes
        )

# TODO: Consider to AVERAGE on the scenario weights for multi-scenario optimization
def add_renewable_penetration_constraint(
    model: Model,
    settings: ProjectParameters,
    sets: xr.Dataset,
    param: xr.Dataset,
    var: Dict[str, linopy.Variable],
    has_battery: bool,
    has_generator: bool,
    has_grid_connection: bool
) -> None:
    """Add renewable penetration constraint with debug logging."""
    years = sets.years.values
    steps = sets.steps.values
    step_duration = settings.advanced_settings.step_duration
    years_steps_tuples = [(years[i] - years[0], steps[i // step_duration]) for i in range(len(years))]

    # Get renewable energy production and curtailment
    total_res_energy_production = var['res_energy_production'].sum(dim=['renewable_sources', 'periods'])
    total_curtailment = var['curtailment'].sum(dim=['renewable_sources', 'periods'])
    
    for year in years:
        step = years_steps_tuples[year - years[0]][1]

        # Renewable energy after curtailment
        yearly_res_production = (total_res_energy_production.sel(steps=step) - total_curtailment.sel(years=year))

        # Initialize total energy production
        yearly_total_production = yearly_res_production

        # Include generator energy if applicable
        if has_generator:
            yearly_generator_production = var['generator_energy_production'].sum(dim=['generator_types', 'periods']).sel(years=year)
            yearly_total_production += yearly_generator_production

        # Include grid imports if applicable
        if has_grid_connection:
            yearly_grid_import = var['energy_from_grid'].sum('periods').sel(years=year)
            yearly_total_production += yearly_grid_import

        # Calculate expected renewable penetration threshold
        min_required_res_production = param['MINIMUM_RENEWABLE_PENETRATION'] * yearly_total_production

        # Add the constraint
        model.add_constraints(
            yearly_res_production >= min_required_res_production,
            name=f"Renewable Penetration Constraint - Year {year}")

    
def add_lost_load_constraint(
    model, settings, sets, param, var,
    has_battery, has_generator, has_compressor, has_grid_connection, has_tes
):
    years = sets.years.values

    for year in years:
        base_electric_demand = param['DEMAND'].sel(years=year)
        lost_load_fraction = param['LOST_LOAD_FRACTION']

        model.add_constraints(
            var['lost_load'].sel(years=year) <= lost_load_fraction * base_electric_demand,
            name=f"Lost Load Constraint - Year {year}"
        )