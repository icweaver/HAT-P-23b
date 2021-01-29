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
        "Na K",
        "Na TiO",
        "K TO",
        "Na K TiO",

    ]:
        for clouds in [False, True]:
            for hazes in [False, True]:
                for heterogeneity in [False, True]:
                    if (not heterogeneity and (clouds and hazes)): continue
#                     print(f"{base_dir=}")
#                     print(f"{fit_R0=}")
#                     print(f"{molecules=}")
#                     print(f"{clouds=}")
#                     print(f"{hazes=}")
#                     print(f"{heterogenity=}")
                    os.system(
                        "python3 opts.py"
                        + f" --base_dir={base_dir}"
                        + f" --spot_temp={spot_temp}"
                        + f" --spot_frac={spot_frac}"
                        + f" --fit_R0={fit_R0}"
                        + f" --molecules {molecules}"
                        + f" --clouds={clouds}"
                        + f" --hazes={hazes}"
                        + f" --heterogeneity={heterogeneity}"
                    )
                    #print(molecules)
                    #exec(open("./opts.py").read())
                    #print(sys.argv)

                    print()
                    i += 1
        #print()

    #print("\n***Next dir***\n")

print(f"Total number of runs: {i}")
