
find Run* -name "*_T[TS]_*_S[13]_V[ST].csv" >csv.list
echo -e "site\tGcm\tWS\tyears\tsdate\tType\tHarvestingYear\tHarvestingDay\tzadok_stage\tstage\tStageName\tSowingYear\tsowing_date\tSowingDay\tSowingDate\tSowingVar\temergence_das\temergence_date\tend_of_juvenile_das\tend_of_juvenile_date\tfloral_initiation_das\tfloral_initiation_date\tflowering_das\tflowering_date\tmaturity_das\tmaturity_date\tgerminationTTTarget\tend_of_juvenileTTTarget\tfloral_initiationTTTarget\tfloweringTTTarget\tmaturityTTTarget\tharvest_ripeTTTarget\tTTAftersowing\tyield\tbiomass\tgrain_no\tgrain_wt\tbarley_incrop_rain\tbarley_incrop_radn\tbarley_incrop_Tmax\tbarley_incrop_Tmin\twatertable\tTheRun\tBarleyType\tNStressZ3\tNodayZ3\tFertSup\tMnTP_Pto_JV1\tMnTP_Pto_JV2\tMnTP_Pto_FIN\tMnTP_Pto_FWR\tMnTP_Pto_GF1\tMnTP_Pto_GF2\tMnTP_Co2_JV1\tMnTP_Co2_JV2\tMnTP_Co2_FIN\tMnTP_Co2_FWR\tMnTP_Co2_GF1\tMnTP_Co2_GF2\tMnOX_Pno_JV1\tMnOX_Pno_JV2\tMnOX_Pno_FIN\tMnOX_Pno_FWR\tMnOX_Pno_GF1\tMnOX_Pno_GF2\tMnOX_Pto_JV1\tMnOX_Pto_JV2\tMnOX_Pto_FIN\tMnOX_Pto_FWR\tMnOX_Pto_GF1\tMnOX_Pto_GF2\tNoD_GS_n_stress_expan\tNoD_GS_n_stress_grain\tNoD_GS_n_stress_pheno\tNoD_GS_n_stress_photo\tNoD_TP_Pto_JV1\tNoD_TP_Pto_JV2\tNoD_TP_Pto_FIN\tNoD_TP_Pto_FWR\tNoD_TP_Pto_GF1\tNoD_TP_Pto_GF2\tNoD_TP_Co2_JV1\tNoD_TP_Co2_JV2\tNoD_TP_Co2_FIN\tNoD_TP_Co2_FWR\tNoD_TP_Co2_GF1\tNoD_TP_Co2_GF2\tNoD_OX_Pto_JV1\tNoD_OX_Pto_JV2\tNoD_OX_Pto_FIN\tNoD_OX_Pto_FWR\tNoD_OX_Pto_GF1\tNoD_OX_Pto_GF2\tDays_JV1\tDays_JV2\tDays_FIN\tDays_FWR\tDays_GF1\tDays_GF2\tNodays_GS\tMnGS_n_stress_expan\tMnGS_n_stress_grain\tMnGS_n_stress_pheno\tMnGS_n_stress_photo\tSWSowing\tCUM_outflow_lat\tCUM_runoff\tCUM_es\tCUM_drain\tSWHarvesting\tGSR\tEPA\tCO2\tTAV\tAMP\tIrrAMT" >combin.result
cat csv.list | while read line
do
	site=`echo $line | awk -F "[/_.]" '{print $3}'`
	gcm=`echo $line | awk -F "[/_.]" '{print $5}'`
	waterlog=`echo $line | awk -F "[/_.]" '{print $6}'`
	years=`echo $line | awk -F "[/_.]" '{print $7}'`
	sdate=`echo $line | awk -F "[/_.]" '{print $8}'`
	type=`echo $line | awk -F "[/_.]" '{print $9}'`

	tail -n +5 $line | awk -v site=$site -v gcm=$gcm -v waterlog=$waterlog -v years=$years -v sdate=$sdate -v type=$type 'BEGIN{OFS="\t"; ORS="\n"} {for (i=5;i<=NF;++i){sum[i]+=$i}colum = NF;row = NR} END{printf("%s\t",site);printf("%s\t",gcm);printf("%s\t",waterlog);printf("%s\t",years);printf("%s\t",sdate);printf("%s\t",type);print $0}' >>combin.result
	
done
sed -i 's/,/\t/g' combin.result
