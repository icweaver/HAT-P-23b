### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 6adcb23e-a0fc-493d-bdac-580c5937cf62
using PlutoUI, Unitful, Measurements, PyCall, NamedArrays, LatexPrint

# ╔═╡ 938f6bdf-ba1d-40ed-b6a2-96f039f81a95
md"""
# Retrieval analysis
"""

# ╔═╡ 65526504-4799-4688-92ee-646a453dc0fa
md"""
## Spot coverage investigation
"""

# ╔═╡ 3200e15f-a15c-4678-a188-3b1b9da31586
md"""
This investigation is to determine whether the features observed in our transmission spectrum originate from the planet's atmosphere, or the star's. If we assume that the variations in photometric activity of HAT-P-23 can be described by an average large spot with temperature ``T_\text{sp}`` and covering fraction ``f_\text{sp}``, the fractional change in apparent luminosity ``L_\text{obs}`` due to the presence of a spot ``(L_\text{obs, sp})`` vs. without ``(L_\text{obs, 0})`` would go like:
"""

# ╔═╡ 139116a6-6123-11eb-27a3-676e43c8eb97
rd(x; digits=3) = round(x; digits=digits)

# ╔═╡ 2a12130e-006e-4b69-8cd5-9ebacfd15378
md"""
### Determining inputs
"""

# ╔═╡ d2c19d43-c50d-4ece-9956-2df6d1d6b080
PlutoUI.Resource("https://user-images.githubusercontent.com/25312320/106031258-11d16680-609d-11eb-940b-237973d7280f.png")

# ╔═╡ e5a80094-7738-4ab8-9a80-409f7cbd8047
md"""
Plugging this and the above parameters in then gives the following bounds:
"""

# ╔═╡ 07685bc2-b93d-456a-8570-6365e21f5bba
const T₀ = 5900u"K" # Effective temperature

# ╔═╡ 7f63268a-f166-438f-9799-3906e38f8aae
ΔL_obs = 0.92 / 0.94 # Observed luminosity (flux) ratio

# ╔═╡ 2515c4e6-cc1a-4561-9adc-1c4c03d8bc77
md"""
From the observed photometric activity below, we estimate that the ratio between the observed flux of HAT-P-23b while its photosphere is spotted (lower shaded region) vs. primarily immaculate (mid-line) is approximately ΔL\_obs = $(round(ΔL_obs; digits=3)) :
"""

# ╔═╡ 2e1dd738-5c42-11eb-0eb6-8b3ee8e736ee
f_sp(T₀, T_sp, ΔL_obs) = T₀^4 * (ΔL_obs - 1.0) / (T_sp^4 - T₀^4)

# ╔═╡ 44fc5ece-6125-11eb-2bf7-a760c99d9bc1
T_sp_upper = 3800u"K"

# ╔═╡ 79c291c3-5112-44da-9f47-a239da53c8ec
f_sp_upper = f_sp(T₀, T_sp_upper, ΔL_obs)

# ╔═╡ 4f29ffd2-6125-11eb-1e1d-21225e9abe13
T_sp_lower = 2200u"K"

# ╔═╡ be042e3d-1fff-4857-b6f1-bed1b1c467b4
f_sp_lower = f_sp(T₀, T_sp_lower, ΔL_obs)

# ╔═╡ f7a04a4f-7a06-4e0c-8e6b-ee0acaa3ea94
md"""
```math
\begin{align}
L_\text{obs} &= \sigma A_\text{s}\left[(1 - f_\text{sp})T_0^4 + f_\text{sp}T_\text{sp}^4\right] \\

\Delta L_\text{obs} &\equiv \frac{L_\text{obs, sp}}{L_\text{obs, 0}}
= \frac{(1 - f_\text{sp})T_0^4 + f_\text{sp}T_\text{sp}^4}{T_0^4}\quad, \\
\end{align}
```

where ``\sigma`` is the Stefan-Boltzmann constant, ``A_\text{s}`` is the total surface area of HAT-P-23, and ``T_\text{0}`` is the effective temperature of HAT-P-23. The corresponding covering fraction is then:

```math
\begin{align}
f_\text{sp} &= \frac{T_0^4}{T_0^4 - T_\text{sp}^4}\left(1 - \Delta L_\text{obs}\right)
\quad.
\end{align}
```

Preliminary atmospheric retrievals hint at the presence of TiO in the transmission spectrum. For a sun-like star $(\text{G0V})$ like HAT-P-23, TiO is stable in the stellar photosphere for temperatures between roughly $2500\text{ K} - 3500 \text{ K}$ (**get source**). We can use this range to bound the predicted spot covering fraction on the star. For,

```math
\Delta L_\text{obs} \equiv \frac{L_\text{obs, sp}}{L_\text{obs, 0}}

= \frac{F_\text{obs, sp}}{F_\text{obs, 0}}

\approx 0.98 \quad,
```

where $F$ is the observed flux, this would correspond to spot covering fractions ranging from
$ $(rd(f_sp_lower)) - $(rd(f_sp_upper)) $.
"""

# ╔═╡ b11e14f3-f1c5-44e3-81b4-d4780dfea172
md"""
## Evidences
"""

# ╔═╡ 907b9ada-09ad-4cc6-8442-d3890248c9ba
md"""
Fixing the spot temperature and covering fraction to each bound then gives the following retrieval results:
"""

# ╔═╡ c1ed0ad2-92ce-4b61-8e8f-a5a77a957eee
const BASE_DIR = "data_retrievals/spot_lower_bound" # spot_upper_bound

# ╔═╡ 32fe7ae2-74a0-454b-a2c1-bda23e2c96b7
const model_types = (
	clear = "HATP23_E1_NoHet_FitP0_NoClouds_NoHaze_fitR0",
	haze = "HATP23_E1_NoHet_FitP0_NoClouds_Haze_fitR0",
	spot = "HATP23_E1_Het_FitP0_NoClouds_NoHaze_fitR0",
	spot_haze = "HATP23_E1_Het_FitP0_NoClouds_Haze_fitR0",
);

# ╔═╡ a627e1dd-70b2-469d-9b9d-af1454cbbdeb
struct Models{T <: Dict{Any, Any}} # Any, Any because Python
	clear::T
	haze::T
	spot::T
	spot_haze::T
end

# ╔═╡ 43e45a72-9191-498f-8bd8-66957e5091c2
struct Retrievals{T <: Models}
	Na::T
	K::T
	TiO::T
	Na_K::T
	Na_TiO::T
	K_TiO::T
	Na_K_TiO::T
end

# ╔═╡ 0776ead9-5ecc-41b9-8a95-fe9dbde2c6bf
const models = collect(fieldnames(Models))

# ╔═╡ eb985e39-b7e2-4717-964f-e8c9f8320fed
const species = collect(fieldnames(Retrievals))

# ╔═╡ 8a03c267-3f64-4768-9aa0-29ca778d0240
begin
	N_species, N_models = length(species), length(models)
	evidences = Array{Measurement{Float64}}(undef, N_species, N_models)
end;

# ╔═╡ 1548b507-2b2b-40bd-b026-50977d38ae8d
Z = NamedArray(
	evidences,
	(species .|> String, models .|> String),
	("species", "models")
)

# ╔═╡ 6049fbf2-ce9c-4b8c-8897-36d2057b0c75
min_evidence, min_evidence_loc = findmin(Z)

# ╔═╡ 2d5bcf3d-3783-4eed-81ea-faf9c783b937
species[min_evidence_loc[1]], models[min_evidence_loc[2]]

# ╔═╡ e3b146b9-29a5-431b-a29b-deba294bde45
ΔlnZ = Z .- min_evidence

# ╔═╡ 5124e5a4-72fc-43f7-acad-09ce03e7ff3b
ΔlnZ_mat = hcat(species, evidences .- min_evidence);

# ╔═╡ 86db95f8-61a7-11eb-14f1-3be866dce482
latex_form

# ╔═╡ 9345710e-2bb7-4dfa-9a27-0e4fc0501eeb
# with_terminal() do
# 	lap(ΔlnZ_mat)
# end

# ╔═╡ 744482f8-3bd4-41ea-863d-2fcd284269d6
# function latex_form(m::Measurement{Float64})
# 	"$(round(m.val; digits=2)) \\pm $(round(m.err; digits=2))"
# end

# ╔═╡ 7b4234f2-57c7-45e6-8360-6e5afb4b7dc5
begin
	py"""
	import pickle
	
	def load_pickle(fpath):
		with open(fpath, "rb") as f:
			data = pickle.load(f)
		return data
	"""
	load_pickle(s) = py"load_pickle"(s)
end

# ╔═╡ f3ed6fb1-3ec5-4aa1-8ea1-ccb186bfab84
retrieval_data = Retrievals(
	(
		Models(
			load_pickle("$(BASE_DIR)/$(model_types.clear)_$(sp)/retrieval.pkl"),
			load_pickle("$(BASE_DIR)/$(model_types.haze)_$(sp)/retrieval.pkl"),
			load_pickle("$(BASE_DIR)/$(model_types.spot)_$(sp)/retrieval.pkl"),
			load_pickle("$(BASE_DIR)/$(model_types.spot_haze)_$(sp)/retrieval.pkl"),
		) for sp in fieldnames(Retrievals)
	)...
);

# ╔═╡ 43ae8062-a4ca-4a9d-99df-8544ebe88772
for (i, model) in enumerate(models)
	for (j, sp) in enumerate(species)
		retr = getfield(getfield(retrieval_data, sp), model)
		evidences[j, i] = retr["lnZ"] ± retr["lnZerr"]
	end
end

# ╔═╡ 509e85b3-be33-4ad2-8070-21501a983723
info(title, text) = Markdown.MD(Markdown.Admonition("correct", title, [text]))

# ╔═╡ 894473ac-f39e-46c5-970b-0c53eb1358e9
info("Inputs",
md"""
|                            | Lower bound               | Upper bound |
|:---------------------------|:-------------------------:|:------------:
| ``\Delta L_\text{obs}``    |   $ $(rd(ΔL_obs)) $       | $ $(rd(ΔL_obs)) $      
| ``T_0\text{ (K)}``         | $ $(rd(T₀.val)) $         | $ $(rd(T₀.val)) $          
| ``T_\text{sp}\text{ (K)}`` | $ $(rd(T_sp_lower.val)) $ | $ $(rd(T_sp_upper.val)) $ 
"""
)

# ╔═╡ b0cad196-a4f0-4245-bbbb-d7fbad621e08
question(text) = Markdown.MD(Markdown.Admonition("note", "Question", [text]))

# ╔═╡ 7055a5db-b068-43cd-9a90-e5c0b06eb270
question(
md"""
Am I estimating stellar variability the correct way?
"""
)

# ╔═╡ Cell order:
# ╟─938f6bdf-ba1d-40ed-b6a2-96f039f81a95
# ╟─65526504-4799-4688-92ee-646a453dc0fa
# ╟─3200e15f-a15c-4678-a188-3b1b9da31586
# ╟─f7a04a4f-7a06-4e0c-8e6b-ee0acaa3ea94
# ╟─894473ac-f39e-46c5-970b-0c53eb1358e9
# ╟─139116a6-6123-11eb-27a3-676e43c8eb97
# ╟─2a12130e-006e-4b69-8cd5-9ebacfd15378
# ╟─2515c4e6-cc1a-4561-9adc-1c4c03d8bc77
# ╟─d2c19d43-c50d-4ece-9956-2df6d1d6b080
# ╟─7055a5db-b068-43cd-9a90-e5c0b06eb270
# ╟─e5a80094-7738-4ab8-9a80-409f7cbd8047
# ╠═07685bc2-b93d-456a-8570-6365e21f5bba
# ╠═7f63268a-f166-438f-9799-3906e38f8aae
# ╠═2e1dd738-5c42-11eb-0eb6-8b3ee8e736ee
# ╠═44fc5ece-6125-11eb-2bf7-a760c99d9bc1
# ╠═79c291c3-5112-44da-9f47-a239da53c8ec
# ╠═4f29ffd2-6125-11eb-1e1d-21225e9abe13
# ╠═be042e3d-1fff-4857-b6f1-bed1b1c467b4
# ╟─b11e14f3-f1c5-44e3-81b4-d4780dfea172
# ╟─907b9ada-09ad-4cc6-8442-d3890248c9ba
# ╠═c1ed0ad2-92ce-4b61-8e8f-a5a77a957eee
# ╠═32fe7ae2-74a0-454b-a2c1-bda23e2c96b7
# ╠═a627e1dd-70b2-469d-9b9d-af1454cbbdeb
# ╠═43e45a72-9191-498f-8bd8-66957e5091c2
# ╠═0776ead9-5ecc-41b9-8a95-fe9dbde2c6bf
# ╠═eb985e39-b7e2-4717-964f-e8c9f8320fed
# ╠═f3ed6fb1-3ec5-4aa1-8ea1-ccb186bfab84
# ╠═8a03c267-3f64-4768-9aa0-29ca778d0240
# ╠═43ae8062-a4ca-4a9d-99df-8544ebe88772
# ╠═1548b507-2b2b-40bd-b026-50977d38ae8d
# ╠═6049fbf2-ce9c-4b8c-8897-36d2057b0c75
# ╠═2d5bcf3d-3783-4eed-81ea-faf9c783b937
# ╠═e3b146b9-29a5-431b-a29b-deba294bde45
# ╠═5124e5a4-72fc-43f7-acad-09ce03e7ff3b
# ╠═86db95f8-61a7-11eb-14f1-3be866dce482
# ╠═9345710e-2bb7-4dfa-9a27-0e4fc0501eeb
# ╠═744482f8-3bd4-41ea-863d-2fcd284269d6
# ╟─7b4234f2-57c7-45e6-8360-6e5afb4b7dc5
# ╟─509e85b3-be33-4ad2-8070-21501a983723
# ╟─b0cad196-a4f0-4245-bbbb-d7fbad621e08
# ╠═6adcb23e-a0fc-493d-bdac-580c5937cf62
