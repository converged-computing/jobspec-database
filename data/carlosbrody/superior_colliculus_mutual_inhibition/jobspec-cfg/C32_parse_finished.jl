include("pro_anti.jl")
# Script for cleaning up good and incomplete fits, moves finished fits to a second directory

# Figure out report filename
reports_dir = "../Reports"
if !isdir(reports_dir); mkdir(reports_dir); end
report_file = reports_dir * "/" * "refit_cleanup"
 
for i=1:5000
    moddex = mod(i,8) + 1;
    if moddex <= 4
        dex1 = "01";
        dex2 = moddex;
    else
        dex1 = "02";
        dex2 = moddex - 4;
    end
    ndex = fld(i-1,8)+1;
    loadname = "../Farms_C32/farm_C32_spock-brody"*dex1*"-0"*string(dex2)*"_"*lpad(string(ndex),4,0)*".jld";
    savename = "../Farms_C32_done/farm_C32_spock-brody"*dex1*"-0"*string(dex2)*"_"*lpad(string(ndex),4,0)*".jld";
      
     # load file
    if isfile(loadname)
        try
        append_to_file(report_file, @sprintf("\n\n**** Checking Farm **** %s ---\n\n", Dates.format(now(), "e, dd u yyyy HH:MM:SS")))
        f = load(loadname)
        
        # check file
        # reasons to keep fitting
        #1. bad hessian
        vals, vecs = eig(f["ftraj3"][2,end])
        goodhess = all(vals .> 0) && isreal(vals);
        if goodhess
           append_to_file(report_file, @sprintf("\n\n**** Hessian looks good **** %s ---\n\n", Dates.format(now(), "e, dd u yyyy HH:MM:SS")))
        else
           append_to_file(report_file, @sprintf("\n\n**** Hessian looks BAD **** %s ---\n\n", Dates.format(now(), "e, dd u yyyy HH:MM:SS")))
        end
        
        #2. Never reach criteria
        finished = haskey(f, "hA")
        if finished
        append_to_file(report_file, @sprintf("\n\n**** Looks like I hit the criteria before **** %s ---\n\n", Dates.format(now(), "e, dd u yyyy HH:MM:SS")))
        else
        append_to_file(report_file, @sprintf("\n\n**** Looks like I did NOT hit the criteria **** %s ---\n\n", Dates.format(now(), "e, dd u yyyy HH:MM:SS")))
        end
        
        # need to refit?
        done_with_fit = finished & goodhess;
        
        # If we are done with fit, copy the file to the good files directory
        if done_with_fit
            cp(loadname, savename)
        end
        end
    end
end
