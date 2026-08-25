"""Small-scene dispersive custom VJP with algebraically reversed CPML."""

from __future__ import annotations

from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_reversible_cpml import (
    cpml_inverse_coefficient_audit,
    update_E_reverse_ADE_with_cpml,
    update_H_reverse_with_cpml,
)


def reversible_ade_cpml_fdtd_prototype(
    *, arrays: Any, objects: Any, config: Any, key: Any
) -> tuple[Any, Any]:
    """Run an O(1)-state ADE+CPML custom VJP without detectors or slicing."""

    import equinox.internal as eqxi
    import jax
    import jax.numpy as jnp
    from fdtdx.fdtd.container import ArrayContainer, FieldState
    from fdtdx.fdtd.forward import forward

    cpml_audit = cpml_inverse_coefficient_audit(objects)
    if cpml_audit["status"] != "PASS":
        raise RuntimeError(f"CPML inverse coefficient audit failed: {cpml_audit}")
    if objects.detectors or arrays.detector_states:
        raise NotImplementedError("CPML prototype does not propagate detectors")
    if arrays.recording_state is not None:
        raise NotImplementedError("CPML prototype does not use a recorder")
    if arrays.electric_conductivity is not None or arrays.magnetic_conductivity is not None:
        raise NotImplementedError("CPML prototype requires conductivity=None")
    if arrays.dispersive_c4 is not None:
        raise NotImplementedError("CPML prototype requires c4=None")
    if arrays.inv_permittivities.shape[0] == 9:
        raise NotImplementedError("CPML prototype rejects full-tensor epsilon")
    if arrays.fields.dispersive_P_curr is None:
        raise ValueError("CPML prototype requires dispersive ADE state")

    base = arrays.reset()
    initial_inv_permittivities = base.initial_inv_permittivities

    def dynamic_tuple(container):
        return (
            container.fields.E,
            container.fields.H,
            container.fields.psi_E,
            container.fields.psi_H,
            container.fields.dispersive_P_curr,
            container.fields.dispersive_P_prev,
            container.inv_permittivities,
            container.inv_permeabilities,
            container.dispersive_c1,
            container.dispersive_c2,
            container.dispersive_c3,
        )

    def from_dynamic(values):
        E, H, psi_E, psi_H, P_curr, P_prev, inv_eps, inv_mu, c1, c2, c3 = values
        return ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
                dispersive_P_curr=P_curr,
                dispersive_P_prev=P_prev,
            ),
            inv_permittivities=inv_eps,
            inv_permeabilities=inv_mu,
            detector_states={},
            recording_state=None,
            electric_conductivity=None,
            magnetic_conductivity=None,
            dispersive_c1=c1,
            dispersive_c2=c2,
            dispersive_c3=c3,
            dispersive_c4=None,
            initial_inv_permittivities=initial_inv_permittivities,
        )

    def one_step(time_step, *values):
        _, output = forward(
            state=(time_step, from_dynamic(values)),
            config=config,
            objects=objects,
            key=key,
            record_detectors=False,
            record_boundaries=False,
            simulate_boundaries=True,
        )
        return dynamic_tuple(output)

    def run_forward(*initial_values):
        def body(state):
            time_step, values = state
            return time_step + 1, one_step(time_step, *values)

        _, final_values = eqxi.while_loop(
            max_steps=config.time_steps_total,
            cond_fun=lambda state: state[0] < config.time_steps_total,
            body_fun=body,
            init_val=(jnp.asarray(0, dtype=jnp.int32), initial_values),
            kind="lax",
        )
        return final_values

    @jax.custom_vjp
    def primitive(E, H, psi_E, psi_H, P_curr, P_prev, inv_eps, inv_mu, c1, c2, c3):
        return run_forward(
            E, H, psi_E, psi_H, P_curr, P_prev, inv_eps, inv_mu, c1, c2, c3
        )

    def primitive_fwd(E, H, psi_E, psi_H, P_curr, P_prev, inv_eps, inv_mu, c1, c2, c3):
        final_values = run_forward(
            E, H, psi_E, psi_H, P_curr, P_prev, inv_eps, inv_mu, c1, c2, c3
        )
        return final_values, final_values

    def primitive_bwd(final_values, final_cotangent):
        def reverse_body(state):
            time_step, current_values, running_cotangent = state
            previous_time = time_step - 1
            current = from_dynamic(current_values)
            previous = update_H_reverse_with_cpml(
                time_step=previous_time,
                arrays=current,
                objects=objects,
                config=config,
            )
            previous = update_E_reverse_ADE_with_cpml(
                time_step=previous_time,
                arrays=previous,
                objects=objects,
                config=config,
            )
            previous_values = dynamic_tuple(previous)
            _, pullback = jax.vjp(
                lambda *values: one_step(previous_time, *values),
                *previous_values,
            )
            return previous_time, previous_values, pullback(running_cotangent)

        _, _, initial_cotangent = eqxi.while_loop(
            max_steps=config.time_steps_total,
            cond_fun=lambda state: state[0] > 0,
            body_fun=reverse_body,
            init_val=(
                jnp.asarray(config.time_steps_total, dtype=jnp.int32),
                final_values,
                final_cotangent,
            ),
            kind="lax",
        )
        return initial_cotangent

    primitive.defvjp(primitive_fwd, primitive_bwd)
    final_values = primitive(*dynamic_tuple(base))
    return jnp.asarray(config.time_steps_total, dtype=jnp.int32), from_dynamic(final_values)
