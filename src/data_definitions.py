genders = {'male': [0, 52],
           'female': [1, 48]}

ages = {'21-35': [0, 60],
        '36-40': [1, 11],
        '41-55': [2, 17],
        '55-99': [3, 3]}

nationalities = {'India'        : [0, 34],
                 'Philippines'  : [1, 31],
                 'Pakistan'     : [2, 6],
                 'BAE'          : [3, 2],
                 'Nepal'        : [4, 2],
                 'Egypt'        : [5, 2],
                 'Afghanistan'  : [6, 1],
                 'UK'           : [7, 1],
                 'Nigeria'      : [8, 1],
                 'SriLanka'     : [9, 1],
                 'Others'       : [10, 19]}

'''markets floor edges [first_market_id, last_market_id]'''
market_ground_floor_edges = [000, 0o07]
market_first_floor_edges  = [108, 142]
market_second_floor_edges = [243, 260]
market_third_floor_edges  = [361, 374]

super_markets = [5]


'''market dict with weights'''
# id : name, weight_21-35, weight_36-40, weight_41-55, weight_55-99
man_markets_ground_floor = {
  000	: ['Cold Stone / Yellow Chili Area' , 0.9862, 0.1849, 0.0493, 0.0123],
  0o01  : ['ENBD Entrance' 				    , 0.8064, 0.3102, 0.0744, 0.0496],
  0o02  : ['Mercato JSYK Lobby' 			, 0.5298, 0.1766, 0.0883, 0.0883],
  0o03  : ['Life Pharmacy Area' 			, 0.6408, 0.3495, 0.1165, 0.0583],
  0o04  : ['Nando\'s and Pancake House Area', 0.4597, 0.6129, 0.3064, 0.1532],
  0o05  : ['Opposite Carrefour Area' 		, 0.2862, 0.2862, 0.4580, 0.1145],
  0o06  : ['Thrifty / Unicare Area' 		, 0.3153, 0.5256, 0.6307, 0.6307],
  0o07  : ['Tim Horton\'s Entrance' 		, 0.0369, 0.0739, 0.1847, 0.0739]
}


man_markets_first_floor = {
       108	    : ['Ahmed Seddiqi Front'                             , 0.0210, 0.0105, 0.0420, 0.0315],
       109	    : ['Ardene / Call it Spring Area'                    , 0.2637, 0.1318, 0.5274, 0.3955],
       110	    : ['Backside of CS Desk Area'                        , 0.0636, 0.0318, 0.1272, 0.0954],
       111	    : ['Cerruti (Instore)'                               , 0.0035, 0.0017, 0.0070, 0.0052],
       112	    : ['Charles Jourdan (Instore)'                       , 0.0013, 0.0007, 0.0026, 0.0020],
       113       : ['ChopardFront'                                   , 0.0048, 0.0024, 0.0096, 0.0072],
       114		: ['Customer Service Area'                           , 0.0575, 0.0287, 0.1150, 0.0862],
       115	    : ['Fontana (Instore)'                               , 0.0010, 0.0013, 0.0007, 0.0003],
       116		: ['Forever 21 Area (Link Area)'                     , 0.0302, 0.0402, 0.0201, 0.0101],
       117		: ['Fountain Area (next to Leonidas)'                , 0.0344, 0.0459, 0.0230, 0.0115],
       118    	: ['Gaudi Accessories (Instore)'                     , 0.0062, 0.0083, 0.0042, 0.0021],
       119       : ['Gaudi Fashion (Instore)'                        , 0.0748, 0.0997, 0.0498, 0.0249],
       120		: ['H&M and Forever 21 Area (Escalator side)'        , 0.5862, 0.7816, 0.3908, 0.1954],
       121       : ['JaferJees Area'                                 , 0.3439, 0.4585, 0.2292, 0.1146],
       122       : ['John Richmond (Instore)'                        , 0.0818, 0.1091, 0.0546, 0.0273],
       123       : ['John Richmond / Charles Jourdan Area'           , 0.4249, 0.5666, 0.2833, 0.1416],
       124       : ['La Vie En Rose Area'                            , 0.0864, 0.1152, 0.0576, 0.0288],
       125       : ['Les Hommes (Instore)'                           , 0.0525, 0.0699, 0.0350, 0.0175],
       126       : ['Main Atrium (Event Area)'                       , 0.3961, 0.5281, 0.2640, 0.1320],
       127       : ['Matalan Area (Lift Side)'                       , 0.3875, 0.2584, 0.1292, 0.0861],
       128       : ['Miniso Area (Link Area)'                        , 0.0372, 0.0248, 0.0124, 0.0083],
       129       : ['Mont Blanc Front'                               , 0.0330, 0.0220, 0.0110, 0.0073],
       130       : ['Piazza Italia (Instore)'                        , 0.1844, 0.1230, 0.0615, 0.0410],
       131       : ['Rivoli Watches Front'                           , 0.0177, 0.0118, 0.0059, 0.0039],
       132	    : ['Rodolfo Zengarini (Instore)'                     , 0.0555, 0.0370, 0.0185, 0.0123],
       133       : ['Shoe Express Area'                              , 0.5764, 0.3843, 0.1921, 0.1281],
       134       : ['Souvenir (Instore)'                             , 0.0171, 0.0114, 0.0057, 0.0038],
       135		: ['Splash Area'                                     , 0.5548, 0.3698, 0.1849, 0.1233],
       136		: ['Starbucks Area (Including Lifts and Escalator)'  , 0.3954, 0.2636, 0.1318, 0.0879],
       137       : ['Store next to Supreme'                          , 0.2174, 0.0836, 0.0268, 0.0067],
       138       : ['The Collection (Instore)'                       , 0.1328, 0.0511, 0.0163, 0.0041],
       139       : ['The Collection and JYSK Area'                   , 0.5001, 0.1923, 0.0616, 0.0154],
       140       : ['The Elements  (Instore)'                        , 0.1091, 0.0420, 0.0134, 0.0034],
       141       : ['Thomas Sabo (Instore)'                          , 0.0520, 0.0061, 0.0018, 0.0012],
       142       : ['Thomas Sabo adjacent(Instore)'                  , 0.0030, 0.0003, 0.0001, 0.0001]
}


man_markets_second_floor = {
    243: ['Anna Virgili (Instore)' 						        , 0.0037, 0.0008, 0.0005, 0.0003],
    244: ['Baldinini Area' 						                , 0.3195, 0.0685, 0.0456, 0.0228],
    245: ['Daiso Area' 						                    , 1.1209, 0.2402, 0.1601, 0.0801],
    246: ['Dino Draghi Area' 						            , 0.6255, 0.1340, 0.0894, 0.0447],
    247: ['Escalator Arae' 						                , 0.4134, 0.0886, 0.0591, 0.0295],
    248: ['Ex-Dome / Kocca / Atrium Area' 					    , 0.1034, 0.0470, 0.0188, 0.0188],
    249: ['Forever21 Front (Link Area)' 						, 0.0439, 0.0199, 0.0080, 0.0080],
    250: ['Gloria Jeans Area (including Escalator and Lifts)'   , 0.1957, 0.0890, 0.0356, 0.0356],
    251: ['Kocca (Instore)' 						            , 0.0036, 0.0016, 0.0007, 0.0007],
    252: ['Main Atrium Area (The Toy Store / Adidas)' 	        , 0.7436, 0.9915, 0.4957, 0.2479],
    253: ['MB Concept (Instore)' 						        , 0.0325, 0.0433, 0.0216, 0.0108],
    254: ['Nila&Nila (Instore)' 						        , 0.0003, 0.0003, 0.0002, 0.0001],
    255: ['Sebago Area' 						                , 0.1957, 0.2610, 0.1305, 0.0652],
    256: ['Sharaf DG / Tasheel Area' 						    , 0.6650, 0.8867, 0.4434, 0.2217],
    257: ['Stadium Area' 						                , 0.3997, 0.5330, 0.2665, 0.1332],
    258: ['The Children\'s Store Area' 						    , 0.1314, 0.1970, 0.6568, 0.3284],
    259: ['Versace Area (Escalator Area)' 					    , 0.0049, 0.0074, 0.0246, 0.0123],
    260: ['Versace Area (ex-Saks Office, Washroom Area)'        , 0.0020, 0.0030, 0.0102, 0.0051]
}

man_markets_third_floor = {
    361				: ['Atrium Area (Panoramic Lift)' 		, 0.8908, 0.2969, 0.1485, 0.1485],
    362				: ['Bombay / Papa John\'s Area' 		, 0.7386, 0.2462, 0.1231, 0.1231],
    363		        : ['Café de Bahia Area' 				, 0.3169, 0.1056, 0.0528, 0.0528],
    364			    : ['Edge - Rope Area (Escalator Area)'  , 0.0829, 0.0276, 0.0138, 0.0138],
    365			    : ['Harmony and Desco Area' 			, 0.1595, 0.0532, 0.0266, 0.0266],
    366             : ['KFC Area' 						    , 0.3433, 0.1561, 0.0624, 0.0624],
    367		        : ['Magic Planet Area' 				    , 0.7386, 0.3357, 0.1343, 0.1343],
    368	            : ['Observatory Area' 					, 0.2539, 0.1154, 0.0462, 0.0462],
    369             : ['Pavilion Garden (Big Screen Area)'  , 0.1760, 0.0800, 0.0320, 0.0320],
    370				: ['Pavilion Garden (Fountain Area)' 	, 0.2156, 0.1797, 0.2515, 0.0719],
    371				: ['Pavilion Garden (Hatam Area)' 		, 0.1715, 0.1430, 0.2001, 0.0572],
    372		        : ['Prime Medical Area' 				, 0.0532, 0.2130, 0.1597, 0.1065],
    373	            : ['Syllogism Area' 					, 0.0013, 0.0052, 0.0039, 0.0026],
    374             : ['Vox Area' 						    , 0.0817, 0.3270, 0.2452, 0.1635]
}

woman_markets_ground_floor = {
 000	: ['Cold Stone / Yellow Chili Area' , 0.9862, 0.1849, 0.0493, 0.0123],
 0o01	: ['ENBD Entrance' 				    , 0.5376, 0.2068, 0.0496, 0.0331],
 0o02	: ['Mercato JSYK Lobby' 			, 0.5298, 0.1766, 0.0883, 0.0883],
 0o03	: ['Life Pharmacy Area' 			, 0.6408, 0.3495, 0.1165, 0.0583],
 0o04	: ['Nando\'s and Pancake House Area', 0.4597, 0.6129, 0.3064, 0.1532],
 0o05	: ['Opposite Carrefour Area' 		, 0.6679, 0.6679, 1.0686, 0.2671],
 0o06	: ['Thrifty / Unicare Area' 		, 0.1351, 0.2252, 0.2703, 0.2703],
 0o07	: ['Tim Horton\'s Entrance' 		, 0.0369, 0.0739, 0.1847, 0.0739]
}


woman_markets_first_floor = {
       108	    : ['Ahmed Seddiqi Front'                            , 0.0315, 0.0157, 0.0630, 0.0472],
       109	    : ['Ardene / Call it Spring Area'                   , 0.3955, 0.1978, 0.7911, 0.5933],
       110	    : ['Backside of CS Desk Area'                       , 0.1908, 0.0954, 0.3816, 0.2862],
       111	    : ['Cerruti (Instore)'                              , 0.0052, 0.0026, 0.0105, 0.0079],
       112	    : ['Charles Jourdan (Instore)'                      , 0.0039, 0.0020, 0.0079, 0.0059],
       113      : ['ChopardFront'                                   , 0.0048, 0.0024, 0.0096, 0.0072],
       114		: ['Customer Service Area'                          , 0.1725, 0.0862, 0.3449, 0.2587],
       115	    : ['Fontana (Instore)'                              , 0.0030, 0.0039, 0.0020, 0.0010],
       116		: ['Forever 21 Area (Link Area)'                    , 0.0905, 0.1207, 0.0603, 0.0302],
       117		: ['Fountain Area (next to Leonidas)'               , 0.1033, 0.1377, 0.0689, 0.0344],
       118    	: ['Gaudi Accessories (Instore)'                    , 0.0187, 0.0249, 0.0125, 0.0062],
       119      : ['Gaudi Fashion (Instore)'                        , 0.0748, 0.0997, 0.0498, 0.0249],
       120		: ['H&M and Forever 21 Area (Escalator side)'       , 0.5862, 0.7816, 0.3908, 0.1954],
       121      : ['JaferJees Area'                                 , 0.8024, 1.0698, 0.5349, 0.2675],
       122      : ['John Richmond (Instore)'                        , 0.0205, 0.0273, 0.0136, 0.0068],
       123      : ['John Richmond / Charles Jourdan Area'           , 0.1062, 0.1416, 0.0708, 0.0354],
       124      : ['La Vie En Rose Area'                            , 0.7778, 1.0371, 0.5186, 0.2593],
       125      : ['Les Hommes (Instore)'                           , 0.0131, 0.0175, 0.0087, 0.0044],
       126      : ['Main Atrium (Event Area)'                       , 0.3961, 0.5281, 0.2640, 0.1320],
       127      : ['Matalan Area (Lift Side)'                       , 0.3875, 0.2584, 0.1292, 0.0861],
       128      : ['Miniso Area (Link Area)'                        , 0.0868, 0.0578, 0.0289, 0.0193],
       129      : ['Mont Blanc Front'                               , 0.0496, 0.0330, 0.0165, 0.0110],
       130      : ['Piazza Italia (Instore)'                        , 0.0615, 0.0410, 0.0205, 0.0137],
       131      : ['Rivoli Watches Front'                           , 0.0118, 0.0079, 0.0039, 0.0026],
       132	    : ['Rodolfo Zengarini (Instore)'                    , 0.0370, 0.0247, 0.0123, 0.0082],
       133      : ['Shoe Express Area'                              , 0.5764, 0.3843, 0.1921, 0.1281],
       134      : ['Souvenir (Instore)'                             , 0.0399, 0.0266, 0.0133, 0.0089],
       135		: ['Splash Area'                                    , 0.8321, 0.5548, 0.2774, 0.1849],
       136		: ['Starbucks Area (Including Lifts and Escalator)' , 0.3954, 0.2636, 0.1318, 0.0879],
       137      : ['Store next to Supreme'                          , 0.0725, 0.0279, 0.0089, 0.0022],
       138      : ['The Collection (Instore)'                       , 0.1087, 0.0418, 0.0134, 0.0033],
       139      : ['The Collection and JYSK Area'                   , 0.7502, 0.2885, 0.0923, 0.0231],
       140      : ['The Elements  (Instore)'                        , 0.0273, 0.0105, 0.0034, 0.0008],
       141      : ['Thomas Sabo (Instore)'                          , 0.0780, 0.0092, 0.0028, 0.0018],
       142      : ['Thomas Sabo adjacent(Instore)'                  , 0.0045, 0.0005, 0.0002, 0.0001]
}

woman_markets_second_floor = {
   243: ['Anna Virgili (Instore)' 						    , 0.0330, 0.0071, 0.0047, 0.0024],
   244: ['Baldinini Area' 						            , 0.4792, 0.1027, 0.0685, 0.0342],
   245: ['Daiso Area' 						                , 1.3700, 0.2936, 0.1957, 0.0979],
   246: ['Dino Draghi Area' 						        , 0.2681, 0.0574, 0.0383, 0.0191],
   247: ['Escalator Arae' 						            , 0.1772, 0.0380, 0.0253, 0.0127],
   248: ['Ex-Dome / Kocca / Atrium Area' 					, 0.4136, 0.1880, 0.0752, 0.0752],
   249: ['Forever21 Front (Link Area)' 						, 0.1316, 0.0598, 0.0239, 0.0239],
   250: ['Gloria Jeans Area (including Escalator and Lifts)', 0.1601, 0.0728, 0.0291, 0.0291],
   251: ['Kocca (Instore)' 						            , 0.0204, 0.0093, 0.0037, 0.0037],
   252: ['Main Atrium Area (The Toy Store / Adidas)' 	    , 0.4957, 0.6610, 0.3305, 0.1652],
   253: ['MB Concept (Instore)' 						    , 0.0108, 0.0144, 0.0072, 0.0036],
   254: ['Nila&Nila (Instore)' 						        , 0.0024, 0.0031, 0.0016, 0.0008],
   255: ['Sebago Area' 						                , 0.5872, 0.7830, 0.3915, 0.1957],
   256: ['Sharaf DG / Tasheel Area' 						, 0.5441, 0.7255, 0.3628, 0.1814],
   257: ['Stadium Area' 						            , 0.2665, 0.3553, 0.1777, 0.0888],
   258: ['The Children\'s Store Area' 						, 0.1314, 0.1970, 0.6568, 0.3284],
   259: ['Versace Area (Escalator Area)' 					, 0.0279, 0.0418, 0.1393, 0.0697],
   260: ['Versace Area (ex-Saks Office, Washroom Area)'     , 0.0115, 0.0173, 0.0576, 0.0288]
}

woman_markets_third_floor = {
   361				: ['Atrium Area (Panoramic Lift)' 		, 1.3361, 0.4454, 0.2227, 0.2227],
   362				: ['Bombay / Papa John\'s Area' 		, 1.1079, 0.3693, 0.1847, 0.1847],
   363		        : ['Café de Bahia Area' 				, 0.4753, 0.1584, 0.0792, 0.0792],
   364			    : ['Edge - Rope Area (Escalator Area)'  , 0.1243, 0.0414, 0.0207, 0.0207],
   365			    : ['Harmony and Desco Area' 			, 0.2392, 0.0797, 0.0399, 0.0399],
   366              : ['KFC Area' 						    , 0.5150, 0.2341, 0.0936, 0.0936],
   367		        : ['Magic Planet Area' 				    , 0.4924, 0.2238, 0.0895, 0.0895],
   368	            : ['Observatory Area' 					, 0.1693, 0.0769, 0.0308, 0.0308],
   369              : ['Pavilion Garden (Big Screen Area)'  , 0.1173, 0.0533, 0.0213, 0.0213],
   370				: ['Pavilion Garden (Fountain Area)' 	, 0.1437, 0.1198, 0.1677, 0.0479],
   371				: ['Pavilion Garden (Hatam Area)' 		, 0.1144, 0.0953, 0.1334, 0.0381],
   372		        : ['Prime Medical Area' 				, 0.0355, 0.1420, 0.1065, 0.0710],
   373	            : ['Syllogism Area' 					, 0.0009, 0.0035, 0.0026, 0.0017],
   374              : ['Vox Area' 						    , 0.0817, 0.3270, 0.2452, 0.1635]
}

#100789
time_of_day = {
    '12AM-8AM' : [0, 0.1204],
    '8AM-11AM' : [1, 0.1573],
    '11AM-1PM' : [2, 0.1348],
    '1PM-4PM'  : [3, 0.1951],
    '4PM-8PM'  : [4, 0.2250],
    '8PM-12PM' : [5, 0.1670],
}