fit1=fullisotropic_fit_exp_
fit2=isotropic_fit_exp_
fit3=anisotropic_fit_exp_

suf=_unconstrained.dat

#for mol in acetone_acetone/; do
for mol in */; do
    dimer=${mol%/}
    echo $dimer
    python plot_compare_total_energies.py  -p $dimer/$fit1 $dimer/$fit2 $dimer/$fit3 -s $suf $suf $suf  -m $dimer
done


