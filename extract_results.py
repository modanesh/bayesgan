import os
import shutil
path = "128"
#for f in os.listdir(path):
a = ["results_100.json","results_200.json","results_300.json","results_400.json","results_500.json","results_600.json","results_700.json","results_800.json","results_900.json","results_1000.json","results_1100.json","results_1200.json","results_1300.json","results_1400.json","results_1500.json","results_1600.json","results_1700.json","results_1800.json","results_1900.json","results_2000.json","results_2100.json","results_2200.json","results_2300.json","results_2400.json","results_2500.json","results_2600.json","results_2700.json","results_2800.json","results_2900.json","results_3000.json","results_3100.json","results_3200.json","results_3300.json","results_3400.json","results_3500.json","results_3600.json","results_3700.json","results_3800.json","results_3900.json","results_4000.json","results_4100.json","results_4200.json","results_4300.json","results_4400.json","results_4500.json","results_4600.json","results_4700.json","results_4800.json","results_4900.json","results_5000.json","results_5100.json","results_5200.json","results_5300.json","results_5400.json","results_5500.json","results_5600.json","results_5700.json","results_5800.json","results_5900.json","results_6000.json","results_6100.json","results_6200.json","results_6300.json","results_6400.json","results_6500.json","results_6600.json","results_6700.json","results_6800.json","results_6900.json","results_7000.json","results_7100.json","results_7200.json","results_7300.json","results_7400.json","results_7500.json","results_7600.json","results_7700.json","results_7800.json","results_7900.json","results_8000.json","results_8100.json","results_8200.json","results_8300.json","results_8400.json","results_8500.json","results_8600.json","results_8700.json","results_8800.json","results_8900.json","results_9000.json","results_9100.json","results_9200.json","results_9300.json","results_9400.json","results_9500.json","results_9600.json","results_9700.json","results_9800.json","results_9900.json","results_10000.json","results_10100.json","results_10200.json","results_10300.json","results_10400.json","results_10500.json","results_10600.json","results_10700.json","results_10800.json","results_10900.json","results_11000.json","results_11100.json","results_11200.json","results_11300.json","results_11400.json","results_11500.json","results_11600.json","results_11700.json","results_11800.json","results_11900.json","results_12000.json","results_12100.json","results_12200.json","results_12300.json","results_12400.json","results_12500.json","results_12600.json","results_12700.json","results_12800.json","results_12900.json","results_13000.json","results_13100.json","results_13200.json","results_13300.json","results_13400.json","results_13500.json","results_13600.json","results_13700.json","results_13800.json","results_13900.json","results_14000.json","results_14100.json","results_14200.json","results_14300.json","results_14400.json","results_14500.json","results_14600.json","results_14700.json","results_14800.json","results_14900.json","results_15000.json","results_15100.json","results_15200.json","results_15300.json","results_15400.json","results_15500.json","results_15600.json","results_15700.json","results_15800.json","results_15900.json","results_16000.json","results_16100.json","results_16200.json","results_16300.json","results_16400.json","results_16500.json","results_16600.json","results_16700.json","results_16800.json","results_16900.json","results_17000.json","results_17100.json","results_17200.json","results_17300.json","results_17400.json","results_17500.json","results_17600.json","results_17700.json","results_17800.json","results_17900.json","results_18000.json","results_18100.json","results_18200.json","results_18300.json","results_18400.json","results_18500.json","results_18600.json","results_18700.json","results_18800.json","results_18900.json","results_19000.json","results_19100.json","results_19200.json","results_19300.json","results_19400.json","results_19500.json","results_19600.json","results_19700.json","results_19800.json","results_19900.json","results_20000.json","results_20100.json","results_20200.json","results_20300.json","results_20400.json","results_20500.json","results_20600.json","results_20700.json","results_20800.json","results_20900.json","results_21000.json","results_21100.json","results_21200.json","results_21300.json","results_21400.json","results_21500.json","results_21600.json","results_21700.json","results_21800.json","results_21900.json","results_22000.json","results_22100.json","results_22200.json","results_22300.json","results_22400.json","results_22500.json","results_22600.json","results_22700.json","results_22800.json","results_22900.json","results_23000.json","results_23100.json","results_23200.json","results_23300.json","results_23400.json","results_23500.json","results_23600.json","results_23700.json","results_23800.json","results_23900.json","results_24000.json","results_24100.json","results_24200.json","results_24300.json","results_24400.json","results_24500.json","results_24600.json","results_24700.json","results_24800.json","results_24900.json","results_25000.json","results_25100.json","results_25200.json","results_25300.json","results_25400.json","results_25500.json","results_25600.json","results_25700.json","results_25800.json","results_25900.json","results_26000.json","results_26100.json","results_26200.json","results_26300.json","results_26400.json","results_26500.json","results_26600.json","results_26700.json","results_26800.json","results_26900.json","results_27000.json","results_27100.json","results_27200.json","results_27300.json","results_27400.json","results_27500.json","results_27600.json","results_27700.json","results_27800.json","results_27900.json","results_28000.json","results_28100.json","results_28200.json","results_28300.json","results_28400.json","results_28500.json","results_28600.json","results_28700.json","results_28800.json","results_28900.json","results_29000.json","results_29100.json","results_29200.json","results_29300.json","results_29400.json","results_29500.json","results_29600.json","results_29700.json","results_29800.json","results_29900.json","results_30000.json","results_30100.json","results_30200.json","results_30300.json","results_30400.json","results_30500.json","results_30600.json","results_30700.json","results_30800.json","results_30900.json","results_31000.json","results_31100.json","results_31200.json","results_31300.json","results_31400.json","results_31500.json","results_31600.json","results_31700.json","results_31800.json","results_31900.json","results_32000.json","results_32100.json","results_32200.json","results_32300.json","results_32400.json","results_32500.json","results_32600.json","results_32700.json","results_32800.json","results_32900.json","results_33000.json","results_33100.json","results_33200.json","results_33300.json","results_33400.json","results_33500.json","results_33600.json","results_33700.json","results_33800.json","results_33900.json","results_34000.json","results_34100.json","results_34200.json","results_34300.json","results_34400.json","results_34500.json","results_34600.json","results_34700.json","results_34800.json","results_34900.json","results_35000.json","results_35100.json","results_35200.json","results_35300.json","results_35400.json","results_35500.json","results_35600.json","results_35700.json","results_35800.json","results_35900.json","results_36000.json","results_36100.json","results_36200.json","results_36300.json","results_36400.json","results_36500.json","results_36600.json","results_36700.json","results_36800.json","results_36900.json","results_37000.json","results_37100.json","results_37200.json","results_37300.json","results_37400.json","results_37500.json","results_37600.json","results_37700.json","results_37800.json","results_37900.json","results_38000.json","results_38100.json","results_38200.json","results_38300.json","results_38400.json","results_38500.json","results_38600.json","results_38700.json","results_38800.json","results_38900.json","results_39000.json","results_39100.json","results_39200.json","results_39300.json","results_39400.json","results_39500.json","results_39600.json","results_39700.json","results_39800.json","results_39900.json","results_40000.json","results_40100.json","results_40200.json","results_40300.json","results_40400.json","results_40500.json","results_40600.json","results_40700.json","results_40800.json","results_40900.json","results_41000.json","results_41100.json","results_41200.json","results_41300.json","results_41400.json","results_41500.json","results_41600.json","results_41700.json","results_41800.json","results_41900.json","results_42000.json","results_42100.json","results_42200.json","results_42300.json","results_42400.json","results_42500.json","results_42600.json","results_42700.json","results_42800.json","results_42900.json","results_43000.json","results_43100.json","results_43200.json","results_43300.json","results_43400.json","results_43500.json","results_43600.json","results_43700.json","results_43800.json","results_43900.json","results_44000.json","results_44100.json","results_44200.json","results_44300.json","results_44400.json","results_44500.json","results_44600.json","results_44700.json","results_44800.json","results_44900.json","results_45000.json","results_45100.json","results_45200.json","results_45300.json","results_45400.json","results_45500.json","results_45600.json","results_45700.json","results_45800.json","results_45900.json","results_46000.json","results_46100.json","results_46200.json","results_46300.json","results_46400.json","results_46500.json","results_46600.json","results_46700.json","results_46800.json","results_46900.json","results_47000.json","results_47100.json","results_47200.json","results_47300.json","results_47400.json","results_47500.json","results_47600.json","results_47700.json","results_47800.json","results_47900.json","results_48000.json","results_48100.json","results_48200.json","results_48300.json","results_48400.json","results_48500.json","results_48600.json","results_48700.json","results_48800.json","results_48900.json","results_49000.json","results_49100.json","results_49200.json","results_49300.json","results_49400.json","results_49500.json","results_49600.json","results_49700.json","results_49800.json","results_49900.json"]
for aa in a:
	print(open(path+"/"+aa).read())
		