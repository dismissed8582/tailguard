# Legacy research-artifact audit

`CVaR_Cox_Reshoring_Extended.pdf` and the images under `results/` are retained
as historical artifacts from the initial repository. They are not executable
specifications, regression fixtures, or evidence for Tailguard's supported
behavior. This note records the concrete reasons.

## Source review

- The PDF labels itself a May 2026 preprint and says it has not undergone peer
  review.
- Reference [3], DOI `10.20944/preprints202601.2128.v2`, identifies a real
  [Preprints.org manuscript](https://www.preprints.org/manuscript/202601.2128/v2/download).
  It is itself a preprint, not a peer-reviewed source.
- Reference [4], *A CVaR-Based Risk Measure for Stochastic Reshoring
  Optimization, Working note (2026)*, supplies no author, publisher, URL, DOI,
  or other identifier. No independently identifiable source was found during
  the July 2026 audit. Claims attributed only to [4] must therefore be treated
  as unverified.
- The PDF's data-availability statement says its results are synthetic. The
  repository does not record the exact scenario sample, complete run
  configuration, dependency and solver versions, or an automated command that
  reproduces its tables and figures. This audit did not reproduce its
  quantitative or runtime claims.

## Worked-example contradiction

Equation (26) gives every sourcing split the loss

\[
Z(x,N)=B(x)+K(x)N,
\]

where `x` lies on a simplex, `B(x)` and `K(x)` are linear in `x`, `N` is one
shared non-negative shock count, and every exposure is non-negative. Translation
invariance and positive homogeneity of CVaR give

\[
\mathbb E[Z]+\lambda\operatorname{CVaR}_\alpha(Z)
=(1+\lambda)B(x)+
\left(\mathbb E[N]+\lambda\operatorname{CVaR}_\alpha(N)\right)K(x).
\]

That objective is linear in `x`. On the stated simplex it has an extreme-point
optimum (one route), except when exact ties create a face of equally optimal
solutions. The interior Cox sourcing splits reported in Section 8.3 therefore
do not follow from the model stated in Equation (26). The current package does
not reproduce or endorse them.

## Result-image status

The inherited notebook contains related plotting routines, but the checked-in
PNGs are not tied to a recorded input snapshot, complete run configuration,
dependency and solver versions, or an automated generation command. Their
exact values and labels therefore cannot be independently reconstructed from
the repository as committed. They must not be used to support claims about
variance reduction, tail dependence, sourcing recommendations, solver scale,
or optimality gaps.

New research results should include the exact input data or a documented public
source, executable generation code, dependency versions, random seeds, and a
machine-checkable validation procedure.

## Artifact preservation

The retained PDF and PNG files remain byte-for-byte unchanged from the initial
repository so their historical provenance can be evaluated in context.
