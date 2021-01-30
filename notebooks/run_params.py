import os
import sys

i = 0
spot_temp = 3800.0
spot_frac = 0.026
for (base_dir, fit_R0) in zip(["spot_upper_fixed_R0", "spot_upper_fit_R0"], [False, True]):
    for molecules in [
        "Na",
        "K",
        "TiO",
        '["Na", "K"]',
        '["Na", "TiO"]',
        '["K", "TiO"]',
        '["Na", "K", "TiO"]',

    ]:
        for clouds in [False, True]:
            for hazes in [False, True]:
                for heterogeneity in [False, True]:
                    if (not heterogeneity and (clouds and hazes)): continue
                    params = {
                        "base_dir":f'"{base_dir}"',
                        "spot_temp":f"{spot_temp}",
                        "spot_frac":f"{spot_frac}",
                        "fit_R0":f"{fit_R0}",
                        "molecules":f'{molecules}',
                        "clouds":f"{clouds}",
                        "hazes":f"{hazes}",
                        "heterogeneity":f"{heterogeneity}",
                    }
                    # Modify opts.py file
                    lines = []
                    with open("opts.py", "r") as f:
                        for line in f.readlines():
                            line = set_param(line, params)
                            lines.append(line)

                    with open("opts.py", "w") as f:
                        for line in lines:
                            f.write(line)

                    # Submit job
                    os.system( # run job script
                        "python3 q.py"
                    )
                    i += 1
                    sys.exit()

print(f"Total number of runs: {i}")
