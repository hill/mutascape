# Controlled benchmark experiment

This experiment follows the restricted-library usefulness study.
That study exposed disconnected candidates and synthetic landscapes with little scope to improve on known singles.
Here the question concerns controlled synthetic benchmarking, rather than transfer to measured method rankings.

Use CreiLOV and eqFP611 singles to construct complete ten-site binary libraries.
Choose sites by increasing coordinate and one substitution per site by alphabetical residue order.
This rule does not select mutations through their fitness.
Enumerate all 1,024 genotypes.
Supply the resulting WT and ten measured singles to every method.

The generator uses its existing linear mode at pair strengths 0, 0.3, 1 and 3.
Five fixed landscape seeds share the same pair field across strengths.
Every landscape preserves every supplied measurement.
No measured combinations select parameters or evaluate this experiment.

Compare random search, additive ridge search, pairwise ridge search and GP expected improvement.
The two ridge methods share regularisation and selection rules.
They differ only in whether their surrogate includes pair indicators.
Use budgets of 32, 64 and 96 evaluations with ten paired repeats.
All neighbourhoods exist because each synthetic library is complete.

Report the exact Walsh variance spectrum, local optimum count and search regret for every condition.
The main contrast is pairwise minus additive ridge regret at 96 evaluations.
Bootstrap intervals resample landscape seeds and paired campaign repeats within each protein.
They describe this synthetic experiment and do not imply biological transfer.

This controlled construction favours a surrogate that can express its pairwise generating function.
Its purpose is to test whether the software exposes the expected failure of a misspecified additive surrogate.
It cannot establish general superiority of pairwise models or realistic absolute combination fitness.
The experiment retains the singles-only input requirement of the original project.
