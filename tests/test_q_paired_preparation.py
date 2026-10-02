"""Single-pass preparation replay against immutable legacy six-slot receipts.

Array receipts were captured before extraction from q_parallel.py SHA256
 d4803257c8c5bd9621b425bfb7c8d1b67a3ff83a653b954938b8fd89830abf40,
using the portable actual-HSX patch (selection and accepted-choice modes).
They cover every PreparedQ array, including physical diffusion coefficients.
"""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from drbx.stencils import q_parallel as prep

LEGACY_HASHES = {'selection:0.0625': {'owners': '659cadd31154cc4c8f793f3582cebc8d8e20dd879a3b5ce28049d523c53e808c',
                      'raw': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf',
                      'raw_to_owner': '6ba049c298f5563f3aa65636efc50fdc25f67f385d3086c4060b9a4b83281d66',
                      'raw_weight': 'cd6283f6ffa67b0e61ec32fdf520080df4a2455c909dfceb371306d96dc6d89e',
                      'owner_raw': '277bac5a3900af17f7d6304f35efb2aba7dc84550eab65300e298d7db8b6a32b',
                      'owner_weight': '692beb835be5ef925677375af98e0a1f3352ef81066b0c7573db109923a33ddd',
                      'slot_points': 'beb52f72b23a0f85422d68be3e08f882cd30ca8ca5713542ecf29604ad01712f',
                      'donor': 'da937361613fa2bfa0572ab953b3d24b18bf23dafaeb66d7c64e3bcce003b5c7',
                      'mask': '288ec7fbe0173302e51ca8fea225c03cba818817191cc58ba7ce289711282a9b',
                      'diffusion_D': '5e25f9cd9c43c941be7e69d909a69a1d5251463c6c321e8be9fc2f4176be9f09',
                      'diffusion_N': '4f6b0921edda72db15f087287a50a4d3b61b88eb68a5702d62dcf31bb57a5619',
                      'boundary_D_node': '88e12f03d6b5a2b46a181424fdb5ab4c6aca6a671fd7adf2186cb56d9e5cb2fe',
                      'boundary_D_tangent': 'a7d33428ef7971390287ad12fd4ccba0d535a97f1118b3f84eb4f136e62b0d67',
                      'boundary_N_normal': '0ea1a3fa3110617dc12e03fd52d55d5f9d29cc61abbb9a4db9b07e164e227b46',
                      'boundary_wall_nodes': '3393711ee2f874729a3b7adc652ca2d6c5894bd0deb569be8555c85e495e723d',
                      'boundary_wall_queries': '7194af10d40d5889b008cf2164300b0fb9733383c32e5506eea3cd006f065a22',
                      'boundary_wall_normal': 'f863d2f11016d3ad09a9d20f4be78f71571f038af3bc5903f174874b8e8e377e',
                      'boundary_value_D_trace': '79840f8cd4edc55268ce8795d46e6e1f4d5d31314be66a5730a01ec90d786793',
                      'boundary_gradient_D_trace': '7b83ff0e137a2bbd71a21f9173b61cfb469ca1adb0096b79173e8422b62b75ea',
                      'boundary_value_N_normal': 'f920b4dae1c3e15feae9d7999f330100dd2af584aa2d8240d08f5a7615d4fd54',
                      'boundary_gradient_N_normal': '2004faa3fd3f88abe811cccad497fbbf4977ee674f4b65de73ea0794af1f4b26',
                      'choice': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                      'indicator': 'b5279919bbc65a8925e261143847667414cace49a538f9c8c439fed72ae465f2',
                      'noise': 'a2a74911242ad617e57800c6abd3ac34105393b3543c2ec0398ad34b6960cddc',
                      'row_value_D': '9268de896f4766d53681f6f856b3c38755e7c2ef08ed80d86eee77b9eb1f4573',
                      'row_gradient_D': 'af28a6310b1253aee8e2e7772736ef19abad3cedfac6c3b3ef24db00c7c5a0b8',
                      'row_value_N': 'febd67f7f61defbcf105841fb45b888227d0acf25548af716d9eae792108a9a4',
                      'row_gradient_N': '3b3539a89ead8cdbe65bd0af0177081e0b94956d9d619e2a8662b8dc73f3d9a1',
                      'magnetic_b': 'df0aec23b4e8cf4a42d86176b79839bf84c5aae14da177669307c1665b1bac36',
                      'magnetic_L': '22d33f2b6bcb5fb1859bb82a65dc1d09a8ca4ba836dce68a988417b132237395',
                      'row_count': '53d5da3628a892eaf4898305014a71d73b6755c9e7af09bc94b7082fa30017f4'},
 'selection:0.03125': {'owners': '659cadd31154cc4c8f793f3582cebc8d8e20dd879a3b5ce28049d523c53e808c',
                       'raw': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf',
                       'raw_to_owner': '6ba049c298f5563f3aa65636efc50fdc25f67f385d3086c4060b9a4b83281d66',
                       'raw_weight': 'cd6283f6ffa67b0e61ec32fdf520080df4a2455c909dfceb371306d96dc6d89e',
                       'owner_raw': '277bac5a3900af17f7d6304f35efb2aba7dc84550eab65300e298d7db8b6a32b',
                       'owner_weight': '692beb835be5ef925677375af98e0a1f3352ef81066b0c7573db109923a33ddd',
                       'slot_points': 'a6d5df90cce7b47ef60e6aa90df2be55faa04f51ec71776c01152c9bbc462424',
                       'donor': 'da937361613fa2bfa0572ab953b3d24b18bf23dafaeb66d7c64e3bcce003b5c7',
                       'mask': '288ec7fbe0173302e51ca8fea225c03cba818817191cc58ba7ce289711282a9b',
                       'diffusion_D': '79f919a4dc238ea0b732ef9eca23de362b23f549a5c4c46832fbeb5d0bf6ed4c',
                       'diffusion_N': 'e37de0c1b23b6d3c429d4ab66024c2177612adaff9affe9a1b3230b7be96aa99',
                       'boundary_D_node': '227203082d13a29297858b7a6790b05c9dc121cfd7c0d06ff9a33e15fa9766e0',
                       'boundary_D_tangent': 'f4a0fb92b26c0c66b64e21be799de1a03ee2cdd381e3bb5080f63d1df4cfe30a',
                       'boundary_N_normal': '45da445c259d98cadb314706c5d9b9de2fe496c32539e681694090cda2e79da8',
                       'boundary_wall_nodes': '3393711ee2f874729a3b7adc652ca2d6c5894bd0deb569be8555c85e495e723d',
                       'boundary_wall_queries': '8829ed890983e876a6afe1a75785356edd935799da1b2cc2b63fa26ab7c5a923',
                       'boundary_wall_normal': 'f863d2f11016d3ad09a9d20f4be78f71571f038af3bc5903f174874b8e8e377e',
                       'boundary_value_D_trace': 'adce49cfabd59127f8886c22b15f8c9a15c90bc957b06c5be23aaa9457bfe5e3',
                       'boundary_gradient_D_trace': '18fbaf3b85bd5d3b29e98a0f9e1ebaccdec83b185be1c87ea06d8ece3565dda0',
                       'boundary_value_N_normal': '2b99374fd39fe6f5552c6b75712cd20b33ad59ae9f426380f6a8ad73967a9de9',
                       'boundary_gradient_N_normal': '67a08f4ffc1d7efba1e2ded8cd592bfe4ebced2f36a3d5162f0f3e88c7b9ba58',
                       'choice': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                       'indicator': 'b5279919bbc65a8925e261143847667414cace49a538f9c8c439fed72ae465f2',
                       'noise': 'a2a74911242ad617e57800c6abd3ac34105393b3543c2ec0398ad34b6960cddc',
                       'row_value_D': '5281c51d38bd14c417c68c85e6bad872b930024b2d06ba07d3dbd638e60739e4',
                       'row_gradient_D': '73f95e0acf6af3af28d7b2f9973cc11176a5c513a4910361f8a4ad84252b7d86',
                       'row_value_N': '0706f634f6da503131b3cfb4cacbbf1fcdf3b8cf5a0f6a7f3d107087996557a3',
                       'row_gradient_N': '44720c9af116c3522fbbfa9c691f81cac86cf14a0e06c2d5a5fbfb6eef7e33b8',
                       'magnetic_b': 'd8d12f51b643ae1907c8653b671c0e7b672d5c8eaf36e8904dead42cd59c79b4',
                       'magnetic_L': '01e0f4d00d073fe30ae2e630680632ab4e0e3512b363869c9849465e7f7af7e4',
                       'row_count': '53d5da3628a892eaf4898305014a71d73b6755c9e7af09bc94b7082fa30017f4'},
 'accepted:0.0625': {'owners': '659cadd31154cc4c8f793f3582cebc8d8e20dd879a3b5ce28049d523c53e808c',
                     'raw': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf',
                     'raw_to_owner': '6ba049c298f5563f3aa65636efc50fdc25f67f385d3086c4060b9a4b83281d66',
                     'raw_weight': 'cd6283f6ffa67b0e61ec32fdf520080df4a2455c909dfceb371306d96dc6d89e',
                     'owner_raw': '277bac5a3900af17f7d6304f35efb2aba7dc84550eab65300e298d7db8b6a32b',
                     'owner_weight': '692beb835be5ef925677375af98e0a1f3352ef81066b0c7573db109923a33ddd',
                     'slot_points': 'beb52f72b23a0f85422d68be3e08f882cd30ca8ca5713542ecf29604ad01712f',
                     'donor': 'da937361613fa2bfa0572ab953b3d24b18bf23dafaeb66d7c64e3bcce003b5c7',
                     'mask': '288ec7fbe0173302e51ca8fea225c03cba818817191cc58ba7ce289711282a9b',
                     'diffusion_D': '5e25f9cd9c43c941be7e69d909a69a1d5251463c6c321e8be9fc2f4176be9f09',
                     'diffusion_N': '4f6b0921edda72db15f087287a50a4d3b61b88eb68a5702d62dcf31bb57a5619',
                     'boundary_D_node': '88e12f03d6b5a2b46a181424fdb5ab4c6aca6a671fd7adf2186cb56d9e5cb2fe',
                     'boundary_D_tangent': 'a7d33428ef7971390287ad12fd4ccba0d535a97f1118b3f84eb4f136e62b0d67',
                     'boundary_N_normal': '0ea1a3fa3110617dc12e03fd52d55d5f9d29cc61abbb9a4db9b07e164e227b46',
                     'boundary_wall_nodes': '3393711ee2f874729a3b7adc652ca2d6c5894bd0deb569be8555c85e495e723d',
                     'boundary_wall_queries': '7194af10d40d5889b008cf2164300b0fb9733383c32e5506eea3cd006f065a22',
                     'boundary_wall_normal': 'f863d2f11016d3ad09a9d20f4be78f71571f038af3bc5903f174874b8e8e377e',
                     'boundary_value_D_trace': '79840f8cd4edc55268ce8795d46e6e1f4d5d31314be66a5730a01ec90d786793',
                     'boundary_gradient_D_trace': '7b83ff0e137a2bbd71a21f9173b61cfb469ca1adb0096b79173e8422b62b75ea',
                     'boundary_value_N_normal': 'f920b4dae1c3e15feae9d7999f330100dd2af584aa2d8240d08f5a7615d4fd54',
                     'boundary_gradient_N_normal': '2004faa3fd3f88abe811cccad497fbbf4977ee674f4b65de73ea0794af1f4b26',
                     'choice': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                     'indicator': 'b5279919bbc65a8925e261143847667414cace49a538f9c8c439fed72ae465f2',
                     'noise': 'a2a74911242ad617e57800c6abd3ac34105393b3543c2ec0398ad34b6960cddc',
                     'row_value_D': '9268de896f4766d53681f6f856b3c38755e7c2ef08ed80d86eee77b9eb1f4573',
                     'row_gradient_D': 'af28a6310b1253aee8e2e7772736ef19abad3cedfac6c3b3ef24db00c7c5a0b8',
                     'row_value_N': 'febd67f7f61defbcf105841fb45b888227d0acf25548af716d9eae792108a9a4',
                     'row_gradient_N': '3b3539a89ead8cdbe65bd0af0177081e0b94956d9d619e2a8662b8dc73f3d9a1',
                     'magnetic_b': 'df0aec23b4e8cf4a42d86176b79839bf84c5aae14da177669307c1665b1bac36',
                     'magnetic_L': '22d33f2b6bcb5fb1859bb82a65dc1d09a8ca4ba836dce68a988417b132237395',
                     'row_count': '53d5da3628a892eaf4898305014a71d73b6755c9e7af09bc94b7082fa30017f4'},
 'accepted:0.03125': {'owners': '659cadd31154cc4c8f793f3582cebc8d8e20dd879a3b5ce28049d523c53e808c',
                      'raw': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf',
                      'raw_to_owner': '6ba049c298f5563f3aa65636efc50fdc25f67f385d3086c4060b9a4b83281d66',
                      'raw_weight': 'cd6283f6ffa67b0e61ec32fdf520080df4a2455c909dfceb371306d96dc6d89e',
                      'owner_raw': '277bac5a3900af17f7d6304f35efb2aba7dc84550eab65300e298d7db8b6a32b',
                      'owner_weight': '692beb835be5ef925677375af98e0a1f3352ef81066b0c7573db109923a33ddd',
                      'slot_points': 'a6d5df90cce7b47ef60e6aa90df2be55faa04f51ec71776c01152c9bbc462424',
                      'donor': 'da937361613fa2bfa0572ab953b3d24b18bf23dafaeb66d7c64e3bcce003b5c7',
                      'mask': '288ec7fbe0173302e51ca8fea225c03cba818817191cc58ba7ce289711282a9b',
                      'diffusion_D': '79f919a4dc238ea0b732ef9eca23de362b23f549a5c4c46832fbeb5d0bf6ed4c',
                      'diffusion_N': 'e37de0c1b23b6d3c429d4ab66024c2177612adaff9affe9a1b3230b7be96aa99',
                      'boundary_D_node': '227203082d13a29297858b7a6790b05c9dc121cfd7c0d06ff9a33e15fa9766e0',
                      'boundary_D_tangent': 'f4a0fb92b26c0c66b64e21be799de1a03ee2cdd381e3bb5080f63d1df4cfe30a',
                      'boundary_N_normal': '45da445c259d98cadb314706c5d9b9de2fe496c32539e681694090cda2e79da8',
                      'boundary_wall_nodes': '3393711ee2f874729a3b7adc652ca2d6c5894bd0deb569be8555c85e495e723d',
                      'boundary_wall_queries': '8829ed890983e876a6afe1a75785356edd935799da1b2cc2b63fa26ab7c5a923',
                      'boundary_wall_normal': 'f863d2f11016d3ad09a9d20f4be78f71571f038af3bc5903f174874b8e8e377e',
                      'boundary_value_D_trace': 'adce49cfabd59127f8886c22b15f8c9a15c90bc957b06c5be23aaa9457bfe5e3',
                      'boundary_gradient_D_trace': '18fbaf3b85bd5d3b29e98a0f9e1ebaccdec83b185be1c87ea06d8ece3565dda0',
                      'boundary_value_N_normal': '2b99374fd39fe6f5552c6b75712cd20b33ad59ae9f426380f6a8ad73967a9de9',
                      'boundary_gradient_N_normal': '67a08f4ffc1d7efba1e2ded8cd592bfe4ebced2f36a3d5162f0f3e88c7b9ba58',
                      'choice': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                      'indicator': 'b5279919bbc65a8925e261143847667414cace49a538f9c8c439fed72ae465f2',
                      'noise': 'a2a74911242ad617e57800c6abd3ac34105393b3543c2ec0398ad34b6960cddc',
                      'row_value_D': '5281c51d38bd14c417c68c85e6bad872b930024b2d06ba07d3dbd638e60739e4',
                      'row_gradient_D': '73f95e0acf6af3af28d7b2f9973cc11176a5c513a4910361f8a4ad84252b7d86',
                      'row_value_N': '0706f634f6da503131b3cfb4cacbbf1fcdf3b8cf5a0f6a7f3d107087996557a3',
                      'row_gradient_N': '44720c9af116c3522fbbfa9c691f81cac86cf14a0e06c2d5a5fbfb6eef7e33b8',
                      'magnetic_b': 'd8d12f51b643ae1907c8653b671c0e7b672d5c8eaf36e8904dead42cd59c79b4',
                      'magnetic_L': '01e0f4d00d073fe30ae2e630680632ab4e0e3512b363869c9849465e7f7af7e4',
                      'row_count': '53d5da3628a892eaf4898305014a71d73b6755c9e7af09bc94b7082fa30017f4'}}

LEGACY_METADATA = {'selection:0.0625': {'campaign_identity': 'c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b',
                      'families': {'ringwise': 1, 'wall': 1},
                      'geometry_identity': 'fixture-HSX',
                      'max_condition': 4.469879843712547,
                      'max_donors': 140,
                      'max_raw_members': 1,
                      'max_reproduction': 6.821210263296962e-13,
                      'n': 32,
                      'n_owner': 25376,
                      'row_diagnostics': [{'condition': 3.3891498406276366,
                                           'family': 'ringwise',
                                           'plane_donors': [],
                                           'plane_exchanges': [],
                                           'plane_expansion': [],
                                           'plane_initial_rank': [],
                                           'plane_rank': [],
                                           'plane_reproduction': [],
                                           'plane_selections': [],
                                           'reproduction': 3.4271323699686902e-15},
                                          {'condition': 4.469879843712547,
                                           'family': 'wall',
                                           'plane_donors': [],
                                           'plane_exchanges': [],
                                           'plane_expansion': [],
                                           'plane_initial_rank': [],
                                           'plane_rank': [],
                                           'plane_reproduction': [],
                                           'plane_selections': [],
                                           'reproduction': 6.821210263296962e-13}],
                      'schema': 'drbx.q-traced-diffusion.v2',
                      'source_identity': 'fixture-traces',
                      'span': 0.0625,
                      'support': 'selective compact28/gradient guard; structured outer; quartic '
                                 'physical wall',
                      'topology_hash': 'b6ca5e96d15ce0165c3d03d215de7a945b627c784702be45636301b0a55d616d:a127c6866fff0e99cde0b1b86c79a97bdf184decd7edfaf5cd540549f6c31088',
                      'trace_hash': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf:29991fb6be3d14ded56b5b07373826346e6d65f0d8eb2ba1dc58eae444cbf556',
                      'trace_method': 'RK4',
                      'trace_steps': 64},
 'selection:0.03125': {'campaign_identity': 'c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b',
                       'families': {'ringwise': 1, 'wall': 1},
                       'geometry_identity': 'fixture-HSX',
                       'max_condition': 4.469879843712547,
                       'max_donors': 140,
                       'max_raw_members': 1,
                       'max_reproduction': 6.821210263296962e-13,
                       'n': 32,
                       'n_owner': 25376,
                       'row_diagnostics': [{'condition': 3.3891498406276366,
                                            'family': 'ringwise',
                                            'plane_donors': [],
                                            'plane_exchanges': [],
                                            'plane_expansion': [],
                                            'plane_initial_rank': [],
                                            'plane_rank': [],
                                            'plane_reproduction': [],
                                            'plane_selections': [],
                                            'reproduction': 3.4271323699686902e-15},
                                           {'condition': 4.469879843712547,
                                            'family': 'wall',
                                            'plane_donors': [],
                                            'plane_exchanges': [],
                                            'plane_expansion': [],
                                            'plane_initial_rank': [],
                                            'plane_rank': [],
                                            'plane_reproduction': [],
                                            'plane_selections': [],
                                            'reproduction': 6.821210263296962e-13}],
                       'schema': 'drbx.q-traced-diffusion.v2',
                       'source_identity': 'fixture-traces',
                       'span': 0.03125,
                       'support': 'selective compact28/gradient guard; structured outer; quartic '
                                  'physical wall',
                       'topology_hash': 'b6ca5e96d15ce0165c3d03d215de7a945b627c784702be45636301b0a55d616d:a127c6866fff0e99cde0b1b86c79a97bdf184decd7edfaf5cd540549f6c31088',
                       'trace_hash': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf:29991fb6be3d14ded56b5b07373826346e6d65f0d8eb2ba1dc58eae444cbf556',
                       'trace_method': 'RK4',
                       'trace_steps': 64},
 'accepted:0.0625': {'campaign_identity': 'c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b',
                     'choice_hash': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                     'choice_provenance': 'literal-baseline',
                     'families': {'ringwise': 1, 'wall': 1},
                     'geometry_identity': 'fixture-HSX',
                     'max_condition': 4.469879843712547,
                     'max_donors': 140,
                     'max_raw_members': 1,
                     'max_reproduction': 6.821210263296962e-13,
                     'n': 32,
                     'n_owner': 25376,
                     'row_diagnostics': [{'condition': 3.3891498406276366,
                                          'family': 'ringwise',
                                          'plane_donors': [],
                                          'plane_exchanges': [],
                                          'plane_expansion': [],
                                          'plane_initial_rank': [],
                                          'plane_rank': [],
                                          'plane_reproduction': [],
                                          'plane_selections': [],
                                          'reproduction': 3.4271323699686902e-15},
                                         {'condition': 4.469879843712547,
                                          'family': 'wall',
                                          'plane_donors': [],
                                          'plane_exchanges': [],
                                          'plane_expansion': [],
                                          'plane_initial_rank': [],
                                          'plane_rank': [],
                                          'plane_reproduction': [],
                                          'plane_selections': [],
                                          'reproduction': 6.821210263296962e-13}],
                     'schema': 'drbx.q-traced-diffusion.v2',
                     'selection_diagnostics': 'not reevaluated; accepted choices reused',
                     'source_identity': 'fixture-traces',
                     'span': 0.0625,
                     'support': 'selective compact28/gradient guard; structured outer; quartic physical '
                                'wall',
                     'topology_hash': 'b6ca5e96d15ce0165c3d03d215de7a945b627c784702be45636301b0a55d616d:a127c6866fff0e99cde0b1b86c79a97bdf184decd7edfaf5cd540549f6c31088',
                     'trace_hash': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf:29991fb6be3d14ded56b5b07373826346e6d65f0d8eb2ba1dc58eae444cbf556',
                     'trace_method': 'RK4',
                     'trace_steps': 64},
 'accepted:0.03125': {'campaign_identity': 'c0d64dba761087a3273d29cd4728689c4f4b5b67f838873a845d1cdb1a809f8b',
                      'choice_hash': '91c3af487e4e61790e4e9daf2840cba24be53247553ba8adf9233692f544a809',
                      'choice_provenance': 'literal-baseline',
                      'families': {'ringwise': 1, 'wall': 1},
                      'geometry_identity': 'fixture-HSX',
                      'max_condition': 4.469879843712547,
                      'max_donors': 140,
                      'max_raw_members': 1,
                      'max_reproduction': 6.821210263296962e-13,
                      'n': 32,
                      'n_owner': 25376,
                      'row_diagnostics': [{'condition': 3.3891498406276366,
                                           'family': 'ringwise',
                                           'plane_donors': [],
                                           'plane_exchanges': [],
                                           'plane_expansion': [],
                                           'plane_initial_rank': [],
                                           'plane_rank': [],
                                           'plane_reproduction': [],
                                           'plane_selections': [],
                                           'reproduction': 3.4271323699686902e-15},
                                          {'condition': 4.469879843712547,
                                           'family': 'wall',
                                           'plane_donors': [],
                                           'plane_exchanges': [],
                                           'plane_expansion': [],
                                           'plane_initial_rank': [],
                                           'plane_rank': [],
                                           'plane_reproduction': [],
                                           'plane_selections': [],
                                           'reproduction': 6.821210263296962e-13}],
                      'schema': 'drbx.q-traced-diffusion.v2',
                      'selection_diagnostics': 'not reevaluated; accepted choices reused',
                      'source_identity': 'fixture-traces',
                      'span': 0.03125,
                      'support': 'selective compact28/gradient guard; structured outer; quartic '
                                 'physical wall',
                      'topology_hash': 'b6ca5e96d15ce0165c3d03d215de7a945b627c784702be45636301b0a55d616d:a127c6866fff0e99cde0b1b86c79a97bdf184decd7edfaf5cd540549f6c31088',
                      'trace_hash': 'c139082fdf842c18cbdec05d317a22805a1b973d1b1da290c41b0f4606f71daf:29991fb6be3d14ded56b5b07373826346e6d65f0d8eb2ba1dc58eae444cbf556',
                      'trace_method': 'RK4',
                      'trace_steps': 64}}

@pytest.fixture(scope='module')
def inputs():
    with np.load(Path(__file__).parent/'data/q05_actual_hsx/patch.npz',allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    m=json.loads(str(a['metadata']))
    t=prep.topology_from_arrays(tuple(a[f'centers_{i}'] for i in range(3)),
                               a['active'],a['aggregate'],a['volume'])
    def geom(p):
        for i in range(m['geom_calls']):
            if np.array_equal(p,a[f'geom_points_{i}']):
                return tuple(a[f'geom_{k}_{i}'] for k in ('J','b','B'))
        raise AssertionError('changed legacy geometry query/batch shape')
    def jac(p):
        for i in range(m['jac_calls']):
            if np.array_equal(p,a[f'jac_points_{i}']):return a[f'jac_J_{i}']
        raise AssertionError('changed legacy wall query')
    return t,a,geom,jac


def paired(inputs,**kw):
    t,a,g,j=inputs
    return prep.prepare_paired_chunks(t,a['owners'],a['ends'],g,j,
        source_identity='fixture-traces',geometry_identity='fixture-HSX',raw=a['raw'],**kw)


@pytest.fixture(scope='module')
def selected(inputs):return paired(inputs)


@pytest.mark.parametrize('mode',['selection','accepted'])
def test_all_arrays_replay_literal_legacy_receipts(inputs,selected,mode):
    pair=selected if mode=='selection' else paired(inputs,
        frozen_choices=selected[0].choice,choice_provenance='literal-baseline')
    for span,q in zip((1/16,1/32),pair):
        receipt=LEGACY_HASHES[f'{mode}:{span}']
        assert set(receipt)==set(prep._ARRAYS)
        for name,expected in receipt.items():
            assert prep.sha256_array(getattr(q,name))==expected,name
        metadata=json.loads(json.dumps({k:v for k,v in q.metadata.items() if not k.endswith('_seconds')},
                                       default=lambda x:x.item() if isinstance(x,np.generic) else x))
        assert metadata==LEGACY_METADATA[f'{mode}:{span}']
        assert q.metadata['span']==span
        assert q.metadata['schema']=='drbx.q-traced-diffusion.v2'
        assert q.metadata['source_identity']=='fixture-traces'
        assert q.metadata['geometry_identity']=='fixture-HSX'
        assert q.metadata['row_diagnostics']==selected[0].metadata['row_diagnostics']


def test_single_magnetic_support_wall_and_projection_pass(inputs,selected,monkeypatch):
    t,a,g,j=inputs
    counts={key:0 for key in ('magnetic','geom','Hybrid','BalancedHybrid','LocalHybrid','wall')}
    original=prep.magnetic_coefficients
    def magnetic(*args):
        counts['magnetic']+=1
        return original(*args)
    monkeypatch.setattr(prep,'magnetic_coefficients',magnetic)
    def geom(points):
        counts['geom']+=1
        return g(points)
    for name in ('Hybrid','BalancedHybrid','LocalHybrid'):
        cls=getattr(prep,name)
        class Counted(cls):
            def rows(self,*args,_name=name,**kwargs):
                counts[_name]+=1
                return super().rows(*args,**kwargs)
        monkeypatch.setattr(prep,name,Counted)
    wall=prep.Wall
    def counted_wall(*args):
        counts['wall']+=1
        return wall(*args)
    monkeypatch.setattr(prep,'Wall',counted_wall)
    outer,inner=paired((t,a,geom,j))
    radial=a['raw']//t.n**2
    interior=int(np.sum(radial<=prep.Hybrid(t).last))
    bulk=int(np.sum((radial>prep.Hybrid(t).last)&(radial<t.n-2)))
    assert counts==dict(magnetic=1,geom=2,Hybrid=interior+bulk,
                       BalancedHybrid=interior,LocalHybrid=interior,
                       wall=int(np.sum(radial>=t.n-2)))
    for name in ('owners','raw','raw_to_owner','raw_weight','owner_raw','owner_weight',
                 'donor','mask','choice','indicator','noise','row_count'):
        assert getattr(outer,name) is getattr(inner,name),name
    np.testing.assert_array_equal(np.sort(outer.owner_raw[outer.owner_weight!=0]),
                                  np.arange(len(outer.raw)))
    np.testing.assert_allclose(outer.owner_weight.sum(1),1,atol=1e-12,rtol=0)


@pytest.mark.parametrize('ai',[0,1])
def test_legacy_wrapper_and_v2_save_load(inputs,selected,tmp_path,ai):
    t,a,g,j=inputs
    span=(1/16,1/32)[ai]
    q=prep.prepare_chunk(t,a['owners'],a['ends'],g,j,span=span,
        source_identity='fixture-traces',geometry_identity='fixture-HSX',raw=a['raw'])
    path=prep.save_chunk(q,tmp_path/'legacy.npz')
    loaded=prep.load_chunk(path,span=span,source_identity='fixture-traces',
                           geometry_identity='fixture-HSX')
    for name,expected in LEGACY_HASHES[f'selection:{span}'].items():
        assert prep.sha256_array(getattr(loaded,name))==expected,name
    assert loaded.metadata==q.metadata
    with pytest.raises(ValueError,match='span'):
        prep.load_chunk(path,span=(1/32 if ai==0 else 1/16))


@pytest.mark.parametrize('case',['empty','duplicate','negative_owner','large_owner','raw','ends','source',
                                 'choices_shape','choices_dtype','choices_provenance',
                                 'orphan_provenance','choices_invalid'])
def test_invalid_inputs(inputs,selected,case):
    t,a,g,j=inputs
    kw=dict(source_identity='fixture-traces',geometry_identity='fixture-HSX',raw=a['raw'])
    owners=a['owners'];ends=a['ends']
    if case=='empty':owners=[]
    elif case=='duplicate':owners=np.repeat(owners,2)
    elif case=='negative_owner':owners=np.array([-1])
    elif case=='large_owner':owners=np.array([len(t.vol)])
    elif case=='raw':kw['raw']=a['raw'][:-1]
    elif case=='ends':ends=ends[:,:3]
    elif case=='source':kw['source_identity']=''
    elif case=='choices_shape':kw.update(frozen_choices=[],choice_provenance='oracle')
    elif case=='choices_dtype':kw.update(frozen_choices=selected[0].choice.astype(float),choice_provenance='oracle')
    elif case=='choices_provenance':kw['frozen_choices']=selected[0].choice
    elif case=='orphan_provenance':kw['choice_provenance']='oracle'
    elif case=='choices_invalid':kw.update(frozen_choices=np.full(len(a['raw']),99),choice_provenance='oracle')
    with pytest.raises(ValueError):
        prep.prepare_paired_chunks(t,owners,ends,g,j,**kw)


def test_reject_incomplete_projection(selected):
    q=selected[0];weights=q.owner_weight.copy();weights[0,0]=0
    with pytest.raises(ValueError,match='projection'):
        replace(q,owner_weight=weights).validate()


# Core-owner control exercises policy paths; magnetic data here are algebraic.
INNER_LEGACY_HASHES = {'selection:0.0625': {'owners': '45adabbfed6d5c15e65388c91d304b353a63747b2e17a7860a030a4b2b26246b',
                      'raw': '231051526353b796dd8f955072532c049c60f9d7c99af7e0bc55bc30ff394c76',
                      'raw_to_owner': '398ff134249c5c1d3e50aacfd66afa8f72fe7ba52e0f6d15a811fcc5ee866359',
                      'raw_weight': '096193a4c52efce44a9f9ac939b29f2d68be5ce1eafe1d8a7cd0998fcf17bc64',
                      'owner_raw': 'b3498bf834b6408e1b3f0b287c495f08345a52ba7ee6392220057d9d7c4e3b4b',
                      'owner_weight': '1a82f2a1a95988a804e6ba6dd6944e1c1ef63905b6937b0c3926cec9b4ff925e',
                      'slot_points': '2cfc70b9da64881548c9454b0c585b96ba4d19e0a2c0a25ceb7154f226744a27',
                      'donor': '39640d88b126025afdade636895bc5a1a86f36fcdbb21a1fc7328f0f6aab9a27',
                      'mask': 'a6b70704acd2aa67adc4e23748c389fa894d1b175f7b6bce2d3fcb341d0d3940',
                      'diffusion_D': '29fa3438935d063fa0cd3bd1fa023f4edd89d9880038a63ca07403ec7d1f9416',
                      'diffusion_N': '29fa3438935d063fa0cd3bd1fa023f4edd89d9880038a63ca07403ec7d1f9416',
                      'boundary_D_node': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                      'boundary_D_tangent': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                      'boundary_N_normal': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                      'boundary_wall_nodes': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                      'boundary_wall_queries': '24b6617e3101cab551fa4a9e7e105b6a1114e97fcad5675b15fa422659c1f760',
                      'boundary_wall_normal': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                      'boundary_value_D_trace': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                      'boundary_gradient_D_trace': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                      'boundary_value_N_normal': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                      'boundary_gradient_N_normal': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                      'choice': '93335d8f30788f171808e4801ef5562691d31b0f35017159f44eaf3e6bde76dd',
                      'indicator': 'c5ebce23e981df436bf543638c2063da6bc9dde4abacf6f61deb44d8dd61afd2',
                      'noise': 'db5e08ed444262c9795a56369d4e0b43ba8af67605681e0bcb78bd2bb11d1aa0',
                      'row_value_D': 'b611f41d180a60dbba583afa2c942da54a9a693be8112b9f8d26aad937f379df',
                      'row_gradient_D': '61d7916aeb714d9616a4b7e638e22e576b2b8b705debe305b992ca05d0998677',
                      'row_value_N': 'b611f41d180a60dbba583afa2c942da54a9a693be8112b9f8d26aad937f379df',
                      'row_gradient_N': '61d7916aeb714d9616a4b7e638e22e576b2b8b705debe305b992ca05d0998677',
                      'magnetic_b': 'fce3ee8fca29cce19b6a072a18b73faa2f06ff6f7a8e5431bab78cba1d08b1c1',
                      'magnetic_L': 'c8478da82937dc5f07d301cb0a06c6aa83f9d1ed7926032e3d8b35c9b2f74edb',
                      'row_count': '8271b28df0033e7029001405607aebd2207dfd4aafe578d9e8ca172b53197bc6'},
 'selection:0.03125': {'owners': '45adabbfed6d5c15e65388c91d304b353a63747b2e17a7860a030a4b2b26246b',
                       'raw': '231051526353b796dd8f955072532c049c60f9d7c99af7e0bc55bc30ff394c76',
                       'raw_to_owner': '398ff134249c5c1d3e50aacfd66afa8f72fe7ba52e0f6d15a811fcc5ee866359',
                       'raw_weight': '096193a4c52efce44a9f9ac939b29f2d68be5ce1eafe1d8a7cd0998fcf17bc64',
                       'owner_raw': 'b3498bf834b6408e1b3f0b287c495f08345a52ba7ee6392220057d9d7c4e3b4b',
                       'owner_weight': '1a82f2a1a95988a804e6ba6dd6944e1c1ef63905b6937b0c3926cec9b4ff925e',
                       'slot_points': '24ca435d4bd307ce89e2f5680b76b1aabf26c23c1208c90e211f50d6fd5cc878',
                       'donor': '39640d88b126025afdade636895bc5a1a86f36fcdbb21a1fc7328f0f6aab9a27',
                       'mask': 'a6b70704acd2aa67adc4e23748c389fa894d1b175f7b6bce2d3fcb341d0d3940',
                       'diffusion_D': '583652025a47a3a6e4a69094990d99ba88467ce7ba65d67d9e08421d82c6b3cc',
                       'diffusion_N': '583652025a47a3a6e4a69094990d99ba88467ce7ba65d67d9e08421d82c6b3cc',
                       'boundary_D_node': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                       'boundary_D_tangent': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                       'boundary_N_normal': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                       'boundary_wall_nodes': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                       'boundary_wall_queries': '24b6617e3101cab551fa4a9e7e105b6a1114e97fcad5675b15fa422659c1f760',
                       'boundary_wall_normal': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                       'boundary_value_D_trace': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                       'boundary_gradient_D_trace': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                       'boundary_value_N_normal': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                       'boundary_gradient_N_normal': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                       'choice': '93335d8f30788f171808e4801ef5562691d31b0f35017159f44eaf3e6bde76dd',
                       'indicator': 'c5ebce23e981df436bf543638c2063da6bc9dde4abacf6f61deb44d8dd61afd2',
                       'noise': 'db5e08ed444262c9795a56369d4e0b43ba8af67605681e0bcb78bd2bb11d1aa0',
                       'row_value_D': '1a6b657ba5ca9f06cc5a18c32d0118a8e36b397bef86a24d6f1e4e7c124fe952',
                       'row_gradient_D': '3ca4fc450b9f931c09c43d879959b820b276a6cc9530412a44ce09981fe2fd9d',
                       'row_value_N': '1a6b657ba5ca9f06cc5a18c32d0118a8e36b397bef86a24d6f1e4e7c124fe952',
                       'row_gradient_N': '3ca4fc450b9f931c09c43d879959b820b276a6cc9530412a44ce09981fe2fd9d',
                       'magnetic_b': 'fce3ee8fca29cce19b6a072a18b73faa2f06ff6f7a8e5431bab78cba1d08b1c1',
                       'magnetic_L': '65ba95ba123675712cdb12b55caba273484ae8b24004892c252f4aa19cc32793',
                       'row_count': '8271b28df0033e7029001405607aebd2207dfd4aafe578d9e8ca172b53197bc6'},
 'accepted:0.0625': {'owners': '45adabbfed6d5c15e65388c91d304b353a63747b2e17a7860a030a4b2b26246b',
                     'raw': '231051526353b796dd8f955072532c049c60f9d7c99af7e0bc55bc30ff394c76',
                     'raw_to_owner': '398ff134249c5c1d3e50aacfd66afa8f72fe7ba52e0f6d15a811fcc5ee866359',
                     'raw_weight': '096193a4c52efce44a9f9ac939b29f2d68be5ce1eafe1d8a7cd0998fcf17bc64',
                     'owner_raw': 'b3498bf834b6408e1b3f0b287c495f08345a52ba7ee6392220057d9d7c4e3b4b',
                     'owner_weight': '1a82f2a1a95988a804e6ba6dd6944e1c1ef63905b6937b0c3926cec9b4ff925e',
                     'slot_points': '2cfc70b9da64881548c9454b0c585b96ba4d19e0a2c0a25ceb7154f226744a27',
                     'donor': '39640d88b126025afdade636895bc5a1a86f36fcdbb21a1fc7328f0f6aab9a27',
                     'mask': 'a6b70704acd2aa67adc4e23748c389fa894d1b175f7b6bce2d3fcb341d0d3940',
                     'diffusion_D': '29fa3438935d063fa0cd3bd1fa023f4edd89d9880038a63ca07403ec7d1f9416',
                     'diffusion_N': '29fa3438935d063fa0cd3bd1fa023f4edd89d9880038a63ca07403ec7d1f9416',
                     'boundary_D_node': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                     'boundary_D_tangent': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                     'boundary_N_normal': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                     'boundary_wall_nodes': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                     'boundary_wall_queries': '24b6617e3101cab551fa4a9e7e105b6a1114e97fcad5675b15fa422659c1f760',
                     'boundary_wall_normal': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                     'boundary_value_D_trace': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                     'boundary_gradient_D_trace': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                     'boundary_value_N_normal': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                     'boundary_gradient_N_normal': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                     'choice': '93335d8f30788f171808e4801ef5562691d31b0f35017159f44eaf3e6bde76dd',
                     'indicator': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                     'noise': '1ef8ef2edcc654aeb0d669651593b81114eabdf723e7bedac817ac22519629fe',
                     'row_value_D': 'b611f41d180a60dbba583afa2c942da54a9a693be8112b9f8d26aad937f379df',
                     'row_gradient_D': '61d7916aeb714d9616a4b7e638e22e576b2b8b705debe305b992ca05d0998677',
                     'row_value_N': 'b611f41d180a60dbba583afa2c942da54a9a693be8112b9f8d26aad937f379df',
                     'row_gradient_N': '61d7916aeb714d9616a4b7e638e22e576b2b8b705debe305b992ca05d0998677',
                     'magnetic_b': 'fce3ee8fca29cce19b6a072a18b73faa2f06ff6f7a8e5431bab78cba1d08b1c1',
                     'magnetic_L': 'c8478da82937dc5f07d301cb0a06c6aa83f9d1ed7926032e3d8b35c9b2f74edb',
                     'row_count': '8271b28df0033e7029001405607aebd2207dfd4aafe578d9e8ca172b53197bc6'},
 'accepted:0.03125': {'owners': '45adabbfed6d5c15e65388c91d304b353a63747b2e17a7860a030a4b2b26246b',
                      'raw': '231051526353b796dd8f955072532c049c60f9d7c99af7e0bc55bc30ff394c76',
                      'raw_to_owner': '398ff134249c5c1d3e50aacfd66afa8f72fe7ba52e0f6d15a811fcc5ee866359',
                      'raw_weight': '096193a4c52efce44a9f9ac939b29f2d68be5ce1eafe1d8a7cd0998fcf17bc64',
                      'owner_raw': 'b3498bf834b6408e1b3f0b287c495f08345a52ba7ee6392220057d9d7c4e3b4b',
                      'owner_weight': '1a82f2a1a95988a804e6ba6dd6944e1c1ef63905b6937b0c3926cec9b4ff925e',
                      'slot_points': '24ca435d4bd307ce89e2f5680b76b1aabf26c23c1208c90e211f50d6fd5cc878',
                      'donor': '39640d88b126025afdade636895bc5a1a86f36fcdbb21a1fc7328f0f6aab9a27',
                      'mask': 'a6b70704acd2aa67adc4e23748c389fa894d1b175f7b6bce2d3fcb341d0d3940',
                      'diffusion_D': '583652025a47a3a6e4a69094990d99ba88467ce7ba65d67d9e08421d82c6b3cc',
                      'diffusion_N': '583652025a47a3a6e4a69094990d99ba88467ce7ba65d67d9e08421d82c6b3cc',
                      'boundary_D_node': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                      'boundary_D_tangent': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                      'boundary_N_normal': 'b6e012910530b69dfae2f922cf33197e2c68683a1c57e92f7ee5d4b2f189293c',
                      'boundary_wall_nodes': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                      'boundary_wall_queries': '24b6617e3101cab551fa4a9e7e105b6a1114e97fcad5675b15fa422659c1f760',
                      'boundary_wall_normal': 'c7c7a03875ab201303edd09b1cfcd7ee1e3c70965d6f7e623c1d3c520680cc79',
                      'boundary_value_D_trace': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                      'boundary_gradient_D_trace': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                      'boundary_value_N_normal': 'fa30a5816c39fb7d080329511b9c3fddf1cdf1263c81a62b9ae96f69f449b807',
                      'boundary_gradient_N_normal': 'e2d31531e85aab156792a96e44730421369caac4df840b53b7a1f40c536e0956',
                      'choice': '93335d8f30788f171808e4801ef5562691d31b0f35017159f44eaf3e6bde76dd',
                      'indicator': 'f2d9170d850d33a41e921ee1b0b4c82f40706bf885d6f0f64ce18632622477d5',
                      'noise': '1ef8ef2edcc654aeb0d669651593b81114eabdf723e7bedac817ac22519629fe',
                      'row_value_D': '1a6b657ba5ca9f06cc5a18c32d0118a8e36b397bef86a24d6f1e4e7c124fe952',
                      'row_gradient_D': '3ca4fc450b9f931c09c43d879959b820b276a6cc9530412a44ce09981fe2fd9d',
                      'row_value_N': '1a6b657ba5ca9f06cc5a18c32d0118a8e36b397bef86a24d6f1e4e7c124fe952',
                      'row_gradient_N': '3ca4fc450b9f931c09c43d879959b820b276a6cc9530412a44ce09981fe2fd9d',
                      'magnetic_b': 'fce3ee8fca29cce19b6a072a18b73faa2f06ff6f7a8e5431bab78cba1d08b1c1',
                      'magnetic_L': '65ba95ba123675712cdb12b55caba273484ae8b24004892c252f4aa19cc32793',
                      'row_count': '8271b28df0033e7029001405607aebd2207dfd4aafe578d9e8ca172b53197bc6'}}

@pytest.mark.parametrize('mode',['selection','accepted'])
def test_complete_core_owner_literal_replay_and_support_pass(inputs,monkeypatch,mode):
    t,_,_,_=inputs
    owners=np.array([0]);raw=prep.raw_members(t,owners)
    points=t.pts[raw];ends=np.repeat(points[:,None,:],4,axis=1)
    ends[:,:,2]+=np.array([-1/16,1/16,-1/32,1/32])*t.g.deta
    queried=[]
    def geom(p):
        queried.append(p.copy())
        return np.ones(len(p)),np.tile([.01,.02,1.],(len(p),1)),np.ones(len(p))
    def jac(p):raise AssertionError('core support unexpectedly requests wall geometry')
    counts={key:0 for key in ('Hybrid','BalancedHybrid','LocalHybrid')}
    for name in counts:
        cls=getattr(prep,name)
        class Counted(cls):
            def rows(self,*args,_name=name,**kwargs):
                counts[_name]+=1
                return super().rows(*args,**kwargs)
        monkeypatch.setattr(prep,name,Counted)
    kw={} if mode=='selection' else dict(frozen_choices=np.zeros(len(raw),np.int8),
                                         choice_provenance='literal-inner-baseline')
    pair=prep.prepare_paired_chunks(t,owners,ends,geom,jac,raw=raw,
        source_identity='algebra-saved-traces',geometry_identity='algebra-constant',**kw)
    assert len(queried)==2
    assert queried[0].shape==(6*len(raw),3)
    assert queried[1].shape==(12*len(raw),3)
    assert counts==dict(Hybrid=len(raw),BalancedHybrid=(len(raw) if mode=='selection' else 0),
                       LocalHybrid=(len(raw) if mode=='selection' else 0))
    for span,q in zip((1/16,1/32),pair):
        assert len(q.raw)==32  # Every member of the agglomerated core owner.
        for name,expected in INNER_LEGACY_HASHES[f'{mode}:{span}'].items():
            assert prep.sha256_array(getattr(q,name))==expected,name
        assert q.metadata['families']=={'repair28':32}
        assert all(row['plane_rank']==[15]*5 for row in q.metadata['row_diagnostics'])
        assert np.all(q.choice==0)
