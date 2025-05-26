import dv_processing as dv

# Open a Davis camera
capture = dv.io.CameraCapture("DAVIS240C_02460013")

# - Access biases raw value
# Photoreceptor bias
default_pr_bp_int = capture.deviceConfigGet(5, 14)
# Source follower bias
default_pr_sf_bp_int = capture.deviceConfigGet(5, 15)
# Differential bias
default_diff_bn_int = capture.deviceConfigGet(5, 10)
# On threshold bias
default_on_bn_int = capture.deviceConfigGet(5, 11)
# Off threshold bias
default_off_bn_int = capture.deviceConfigGet(5, 12)
# Refractory period bias
default_refr_bp_int = capture.deviceConfigGet(5, 16)


print("Default photoreceptor bias:", default_pr_bp_int)
print("Default source follower bias:", default_pr_sf_bp_int)
print("Default differential bias:", default_diff_bn_int)
print("Default on threshold bias:", default_on_bn_int)
print("Default off threshold bias:", default_off_bn_int)
print("Default refractory period bias:", default_refr_bp_int)

print("Default photoreceptor bias pr_bp (bin):", bin(default_pr_bp_int))
print("Default source follower bias sf_bp (bin):", bin(default_pr_sf_bp_int))
print("Default differential bias diff_bn (bin):", bin(default_diff_bn_int))
print("Default on threshold bias on_bn (bin):", bin(default_on_bn_int))
print("Default off threshold bias off_bn (bin):", bin(default_off_bn_int))
print("Default refractory period bias refr_bp (bin):", bin(default_refr_bp_int))

# class CaerBiasCoarseFine:
#     # Coarse current, from 0 to 7, creates big variations in output current.
#     coarse_value: int
#     # Fine current, from 0 to 255, creates small variations in output current.
#     fine_value: int
#     # Whether this bias is enabled or not.
#     enabled: bool
#     # Bias sex: true for 'N' type, false for 'P' type.
#     sex_n: bool
#     # Bias type: true for 'Normal', false for 'Cascode'.
#     type_normal: bool
#     # Bias current level: true for 'Normal, false for 'Low'.
#     current_level_normal: bool


# def caer_bias_coarse_fine_generate(all_bias_values: CaerBiasCoarseFine) -> int:
#     """
#     Generate the actual bias integer value to be passed to the device from the different bias components.
#     :param all_bias_values: structure containing the different bias value components (see 'CaerBiasCoarseFine' class)
#     :return: the bias value as integer to be passed to the device
#     """
#     bias_value = 0
#     # Build up bias value from all its components.
#     if all_bias_values.enabled:
#         bias_value |= 0x01
#     if all_bias_values.sex_n:
#         bias_value |= 0x02
#     if all_bias_values.type_normal:
#         bias_value |= 0x04
#     if all_bias_values.current_level_normal:
#         bias_value |= 0x08

#     bias_value = bias_value | ((all_bias_values.fine_value & 0xFF) << 4)
#     bias_value = bias_value | ((all_bias_values.coarse_value & 0x07) << 12)

#     return bias_value


# def caer_bias_coarse_fine_parse(bias_value: int) -> CaerBiasCoarseFine:
#     """
#     Extracts the different bias value components from the bias value integer as stored on the device.
#     :param bias_value: the bias value as integer as stored on the device
#     :return: the structure containing the different bias value components (see 'CaerBiasCoarseFine' class)
#     """
#     # Decompose bias integer into its parts.
#     all_bias_values = CaerBiasCoarseFine()
#     all_bias_values.coarse_value = bias_value & 0x01
#     all_bias_values.fine_value = bias_value & 0x02
#     all_bias_values.enabled = bool(bias_value & 0x04)
#     all_bias_values.sex_n = bool(bias_value & 0x08)
#     all_bias_values.type_normal = bool((bias_value >> 4) & 0xFF)
#     all_bias_values.current_level_normal = bool((bias_value >> 12) & 0x07)
#     return all_bias_values


# # - Change biases values
# # Convert bias integer to values
# coarse_fine_pr_bp = caer_bias_coarse_fine_parse(default_pr_bp_int)
# coarse_fine_pr_sf_bp = caer_bias_coarse_fine_parse(default_pr_sf_bp_int)
# coarse_fine_diff_bn = caer_bias_coarse_fine_parse(default_diff_bn_int)
# coarse_fine_on_bn = caer_bias_coarse_fine_parse(default_on_bn_int)
# coarse_fine_off_bn = caer_bias_coarse_fine_parse(default_off_bn_int)
# coarse_fine_refr_bp = caer_bias_coarse_fine_parse(default_refr_bp_int)
# # For example here, add 1 on the log-scale coarse value, i.e. multiply bias value by 10 (approximately)
# coarse_fine_pr_bp.coarse_value += 1
# coarse_fine_pr_sf_bp.coarse_value += 1
# coarse_fine_diff_bn.coarse_value += 1
# coarse_fine_on_bn.coarse_value += 1
# coarse_fine_off_bn.coarse_value += 1
# coarse_fine_refr_bp.coarse_value += 1
# # Convert back
# new_pr_bp_value = caer_bias_coarse_fine_generate(coarse_fine_pr_bp)
# new_pr_sf_bp_value = caer_bias_coarse_fine_generate(coarse_fine_pr_sf_bp)
# new_diff_bn = caer_bias_coarse_fine_generate(coarse_fine_diff_bn)
# new_on_bn = caer_bias_coarse_fine_generate(coarse_fine_on_bn)
# new_off_bn = caer_bias_coarse_fine_generate(coarse_fine_off_bn)
# new_refr_bp = caer_bias_coarse_fine_generate(coarse_fine_refr_bp)

# # - Set biases raw value
# # Setting photoreceptor bias
# capture.deviceConfigSet(5, 14, new_pr_bp_value)
# # Setting source follower bias
# capture.deviceConfigSet(5, 15, new_pr_sf_bp_value)
# # Setting differential bias
# capture.deviceConfigSet(5, 10, new_diff_bn)
# # Setting on threshold bias
# capture.deviceConfigSet(5, 11, new_on_bn)
# # Setting off threshold bias
# capture.deviceConfigSet(5, 12, new_off_bn)
# # Setting refractory period bias
# capture.deviceConfigSet(5, 16, new_refr_bp)