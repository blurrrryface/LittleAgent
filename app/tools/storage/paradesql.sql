CALL paradedb.create_bm25(
        index_name => 'knowledge_base_bm25',
        schema_name => 'document',
        table_name => 'knowledge_base',
        key_field => 'id',
        text_fields => '{
    document_content: {tokenizer: {type: "chinese_lindera"}}, title: {tokenizer: {type: "chinese_lindera"}}
  }'
     );


CREATE INDEX ON document.knowledge_base
    WITH (m = 16, ef_construction = 64);


select
    t1.*
from document.knowledge_base t1
where t1.id in (
    SELECT id
    FROM knowledge_base_bm25.rank_hybrid(
        bm25_query => 'document_content:"who am i?"',
        similarity_query => '''[0.01912241377993679, -0.03760205853837174, -0.0709457602021006, 0.03014994145832711, 0.004072551972375224, -0.013427871851936151, -0.020126742704339798, -0.005523806755910169, -0.016189774065738047, -0.03250006871999152, -0.04447166502852713, 0.03975132001515554, -0.0038164482829169677, 0.011770729592332463, 0.009571250272344681, -0.023220074673914005, -0.04262370129774168, -0.04178006462871413, -0.003394630181233832, 0.011529691209269272, -0.014904234160089258, 0.04712309189883499, 0.04290491227898746, 0.01949401520256914, 0.03117435528483759, 0.02171358221863204, 0.0483282866081186, 0.014753585287090082, 0.038485868736904426, 0.04455201208753741, -0.0012635707051925791, -0.04101677501869686, 0.019242934368452213, -0.0140003390594491, -0.05652360467078283, 0.03647721088809841, -0.025771067720459014, -0.05833139859735334, -0.0736373643272038, -0.08323874102138715, -0.010008133680422882, -0.008998783064831732, 0.04230231678699076, -0.04925226661086623, 0.06821398999807265, 0.0010997396599938913, -0.015346139259355601, 0.019845529860448918, -0.008893328481203288, 0.013648823470246773, -0.003218872852293943, -0.011087786110002927, 0.02524881742482749, 0.07536481119476404, -0.038285001089378726, -0.008752722990580396, 0.053671313878239466, 0.053711487407744604, -0.005212464696212981, 0.021271677119365697, -0.019052111965947895, -0.027157041449870654, 0.017495402598784508, 0.018379210934672097, -0.0473239558210705, -0.052626813286976416, 0.011660253317515877, 0.004863460418266, -0.022034966729382963, 0.0350711485312891, -0.009395492943404792, 0.013026140282174949, 0.015707697299611665, -0.012473759839414568, 0.05077884583090076, 0.02504795163994689, -0.036919115987364755, 0.03754179824411403, -0.04587772366004624, 0.02790024429513535, 0.012664582241918885, 0.0075726367372375, -0.01085679110931602, -0.01860016348430527, 0.005282767907185702, -0.04149885364746835, -0.007914108012740994, 0.0047981793641426965, 0.031676518815716544, -0.013749256362687074, -0.01969488098744974, -0.0046174003440146655, -0.06857555176361892, -0.00980726789554228, -0.021894361238760072, -0.017194102990141055, 0.00033801929936909, -0.01324709283180812, -0.03844569520739929, -0.010324496499985661, 0.04668118679956865, -0.013076356262733823, 0.03663790128082877, 0.03792344304912266, -0.04310577620122296, -0.020990464275474817, -0.03577418157233885, 0.03370526342927513, -0.028643447140400048, 0.03806405040239065, -0.05114040759644702, 0.0017663626459619634, -0.03651738441760355, -0.031656432050963974, 0.0055037199911575985, 0.01979531481121259, 0.00037411235553790826, -0.007085537348600424, 0.016792371420379853, -0.021572974865364053, 0.029848641849683655, -0.0241641433040593, 0.04178006462871413, -0.00453705375066566, -0.028301975864896554, -0.007427009089765191, 0.048649671118869524, 0.043587854829994443, -0.014181118079577133, -0.021010551040227386, -0.041297985999942645, -0.04099668825394429, -0.015095056562593575, -0.01004328458741733, -0.013237049449431834, 0.0031812104012135115, -0.04877019170738495, 0.010274280519426787, -0.0016684407155310526, 0.014964494454346968, 0.012785101899111756, -0.005373157417249717, 0.009104237648460174, 0.0027518599955787993, -0.013809516656944784, 0.0050718587399288155, 0.015386311857538191, 0.004567183897794515, 0.010143717479857631, -0.004105192499436876, 0.05174300308844373, 0.02661470438948656, -0.04258352776823654, -0.03420742882279918, -0.003741123613586742, -0.014221291609082273, 0.0011807136154954585, -0.003954543160776425, -0.02659461762473399, 0.0458375501305411, -0.0008222939576090065, -0.000789025544525859, 0.020217131283081266, 0.011419214934452686, -0.02390301722492099, 0.030089681164069397, 0.055599624668035204, -0.04178006462871413, -0.009244843139083065, -0.0032841541392478837, -0.01453263273745691, 0.07869917689078869, -0.006909780019660534, -0.03374543695878027, -0.040896254430181446, 0.010445016157178532, 0.014693325924154921, -0.05222508171721521, -0.055117546039263714, -0.018238605444049205, 0.027116867920365516, 0.010455059539554818, -0.0014688303534474867, -0.015044840582034699, 0.015928649849244837, -0.03195773352225252, 0.016239990977619474, 0.00501159891133238, 0.0037135045448825955, 0.06491979038097276, -0.007140775486008717, 0.020709253294229034, 0.0126545388595426, -0.015014710434905845, -0.013377654940054726, -0.03336379215377164, 0.011338868806764955, 0.001018137993093806, 0.0054133304810935826, 0.03999236119218639, 0.0008022073674794145, -0.008843112500644413, 0.016300251271877186, -0.030170028223079678, 0.003419738404343907, 0.03714006667435282, 0.013819560039321069, 0.030752536950323813, 0.008722592843451542, -0.009566228581156539, 0.06122386291940184, -0.008114973797621595, 0.010153760862233916, 0.00832588296487848, -0.024867172619818856, 0.03722041373336311, 0.05097971347842646, -0.02251704349550935, -0.05186352367695915, 0.018439471228929805, -0.020026308880576948, -0.0012648261279896147, 0.028482754885024587, -0.020809685255346787, -0.03892777011088057, 0.005488654917593171, -0.04087616766542888, -0.008295752817749627, 0.01647098690962893, 0.017856960639040572, -0.007271338059916597, 0.0011097829259548576, -0.011971595377213064, -0.04868984464837466, -0.02994907567344651, -0.0029150633293789693, -0.029386651848309842, 0.02139219584523602, 0.00834094803844291, 0.024686393599690826, 0.05049763484965498, -0.004627443260729676, -0.009993068606858455, 0.000917077447462647, -0.011379041404947545, 0.011218349149572085, -0.010063371352169899, 0.05652360467078283, -0.008441380930883209, 0.04487339659828834, 0.055197893098274, 0.03266076283801208, 0.005438438471373021, 0.00250579945566619, -0.021914448003512645, -0.004316101666693763, 0.058371572126858486, 0.016722069606390957, 0.06226836351066489, -0.04455201208753741, -0.007557571663673072, 0.01997609383134062, -0.021934532905620113, -0.028482754885024587, 0.013568477342559043, -0.010786487432682026, -0.008918436937144001, -0.0020350205633905766, -0.008260601910755177, -0.00698008276497198, -0.020990464275474817, -0.008456446004447636, 0.019242934368452213, 0.00939047125221665, 0.0009685492582721736, -0.0368186821636019, 0.013056270429303803, 0.02161314839486919, -0.06142473056692754, 0.019725011134578595, -0.039490195798662336, -0.009300081742152634, -0.042503180709226256, -0.02781989909877017, -0.015215576219786446, -0.015828216025481987, 0.002438007323118178, 0.015657480387730238, 0.06945935451157122, -0.04238266012071084, -0.030511499498583172, -0.003660777253068374, -0.044993917186803754, -0.02663479115423913, 0.03129487401070791, -0.0017475315368370664, 0.005453503544937449, -0.04567685973781074, -0.039490195798662336, 0.03033072047845514, 0.011047613511820337, 0.047002571310319576, 0.005895408178542517, 0.023220074673914005, 0.02366197791053525, -0.043748548948015005, 0.031033749794214697, 0.017063540881894452, 0.02271790928038995, -0.022115313788393244, -0.010947180619380036, -0.048408633667128886, 0.021934532905620113, -0.04559651267880046, -0.03428777588180946, 0.0032515133793555944, -0.05415339250701095, 0.005905451560918802, 4.710241557052866e-06, -0.030170028223079678, -0.030933317833096944, -0.02014682946909237, -0.004466751005354214, 0.02506803840469946, -0.011830988955267624, 0.024405182618445043, -0.014060598422384262, 0.0067340222250593705, -0.0088180040447037, -0.01931323618244111, 0.009626487944091698, -0.04262370129774168, 0.022155485455253286, -0.005694542393661915, 0.04551616561979018, -0.047042744839824714, -0.029587517633190445, 0.04648032287733315, 0.01684258833226128, 0.03338387891852421, -0.0137994732745685, 0.005059304511958459, 0.006402593866270888, 0.009204670540900475, -0.04330664384874866, 0.03033072047845514, 0.03061193145970092, -0.008963631226514734, -0.0020563625181095452, 0.037039632850589976, -0.004208136237471249, 0.014030469206577956, 0.04800690209736768, 0.026072367329102467, 0.017394968775021658, -0.011318782042012384, 0.0073466629620774606, -0.005759823913446493, 0.02388293046016842, 0.024425267520552515, 0.013578520724935327, 0.01057557919674769, -0.00396458654315271, 0.02131185064887084, -0.014140944550071992, 0.015828216025481987, 0.008089866273003432, -0.03430785892127183, 0.012011768906718205, 0.03553314039530801, -0.058893820559844906, 0.03231929156250859, -0.018499729660542415, -0.06668741077803812, 0.029266131259794423, -0.025650548994588695, 0.01530596572985046, 0.017033410734765594, -0.02418423006881187, 0.01518544607265759, -0.047725687390831696, -0.020910119079109637, -0.012935750772110932, -0.041137295607212285, -0.007748394066177389, -0.005099478041463599, 0.012132287632588527, -0.0035553229022705682, 0.00784882695861769, -0.033805697253037975, 0.04238266012071084, -0.007070472275035996, -0.027538686254879288, 0.002538440215558479, 0.0027443274587965853, -0.03981158030941326, 0.02757885978438443, 0.009772116057225283, -0.062067499588429394, -0.020648992999971322, 0.037421277655598606, 0.01822856206167292, -0.028442583218164546, -0.027438254293761538, 0.003703461162506311, -0.0021580508333468814, 0.008145103944750449, 0.006176620091110849, 0.010028219513852903, 0.03294197381925786, -0.004805711900924911, -0.030511499498583172, 0.03788326951961752, 0.029527257338932734, 0.02922595773028928, -0.014733498522337511, -0.009224757305653046, 0.012815231114918061, -0.029788381555425947, -0.05339010289699368, 0.04856932405985925, 0.0041127250362190905, -0.04471270248026778, 0.00750233352626478, -0.009079129192519462, -0.014341810334952593, 0.0008084844232569332, -0.04133815952944779, 0.0782171019873074, 0.0028698685743469617, 0.09987042577432684, -0.002330042126726301, -0.07544515825377432, 0.006729000533871228, -0.04051460962517281, -0.03252015548474409, -0.0010765145710793695, -0.009465795688716238, -0.01860016348430527, -0.011700426847021017, 0.02504795163994689, 0.013397741704807297, 0.018971764906937614, -0.06029988291665422, -0.03171669234522168, -0.009064064118955035, -0.006076187198670548, 0.003552812056676497, -0.007863892032182117, -0.03149574165823361, -0.024907346149323998, -0.036396863829088126, -0.003889262106653122, 0.008029605978745722, -0.023802585263803237, -0.05969728369936731, 0.03205816362072517, 0.015145273474475, -0.04925226661086623, -0.03780292246060724, -0.022235832514263566, -0.034127081763788904, -0.03908846422890113, -0.011750642827579894, 0.03450872656879753, 0.010073414734546185, 0.017063540881894452, 0.014090728569513118, 0.03493054490331131, -0.015356182641731886, -0.03647721088809841, 0.03878716648290278, -0.005624239648350469, 0.004527010368289374, -0.023621806243675207, 0.06463858312501718, -0.034810024314795886, 0.03205816362072517, 0.003994716690281565, -0.043145949730728106, -0.03396638764576834, 0.014582849649338335, 0.012955837536863503, -0.0054484818537493054, -0.0657634270500003, 0.10902990109453402, -0.01261436533003746, 0.003605539115660081, -0.02392310398967356, -0.015607264407171363, -0.025911675073727006, 0.0006923589204766652, 0.061705941548173326, -0.005538871363813321, -0.01010354488167504, 0.03977140677990811, -0.03720032696861054, 0.026072367329102467, -0.0002634792983908414, -0.021874274474007502, -0.038465781972151857, -0.050899366419416184, -0.014663195777026065, 0.01675219975351981, 0.01075635821687572, 0.011951508612460495, -0.016631679165004392, -0.04238266012071084, 0.04290491227898746, 0.03348431274228705, 0.014331766952576309, -0.01445228660976918, 0.0012535274392316128, -0.0006659953327772137, -0.005212464696212981, -0.007140775486008717, 0.010113587332728775, 0.012212633760276257, 0.008250558528378893, -0.012413499545156859, 0.017846917256664287, -0.002410388254414032, 0.012282937436910252, -0.008366056494383621, -0.020849858784851925, -0.014221291609082273, 0.001530345488425639, 0.01712380117615216, -0.03302232087826814, 0.03472967725578561, 0.006342334503335728, 0.0020915141235959053, 0.030632018224453494, -0.03274110989702236, 0.04487339659828834, 0.015456615534172187, -0.022898688300517982, -0.02263756408402477, 0.03852603854111936, 0.017435142304526797, -0.0035201712967842078, 0.017053497499518164, -0.027136954685118085, 0.007723286075897951, -0.02808102331526338, -0.002736795154845009, -0.033223184800503645, -0.046882054447094354, -0.0019082241414584838, 0.026012107034844755, 0.00765298286492523, 0.012754971751982902, 0.0665668901895227, 0.006739043916247514, 0.016772284655627284, -0.00442908878710442, -0.017354795245516516, 0.004788135981766412, 0.032198770973993165, -0.02384275879330838, 0.01714388794090473, -0.0018981807590821988, 0.013588564107311614, -0.00689973663728425, 0.014201204844329702, 0.03332361862426649, -0.009495925835845093, 0.025509943503965803, 0.015526918279483633, 0.006181641782298991, -0.02757885978438443, -0.007336619579701176, -0.0377627489311021, 0.01126856513013096, 0.002997920535491409, -0.006492983376334904, 0.07809658139879198, 0.017153929460635917, -0.023862843695415848, -0.026233059584477927, -0.0019722501220307076, -0.06512065802849845, 0.005134629414119322, -0.007592723036328795, -0.011198262384819514, -0.02512829869895717, 0.0065532432049313395, -0.011650209935139592, -0.011730556994149873, 0.026956175664990055, 0.0021706048284866005, 0.015175403621603856, -0.03147565489348104, -0.03243980842573381, 0.00915445362901905, -0.027639120078642138, 0.003997227535875636, 0.0014851507333936312, 0.007211078231320162, -0.018690552063046734, -0.007773502522118101, 0.008677397622758258, -0.010118609023916918, 0.023139727614903724, -0.025429596444955523, -0.04398958639975565, -0.039570542857672614, -0.003992205844687494, 0.011720513611773588, -0.015356182641731886, -0.0019044578730673769, 0.018459557993682377, -0.02669505144849684, -0.06724983274052969, -0.020568647803606142, 0.04190058521722955, -0.008637225024575668, 0.0169631070581316, -0.027016435959247763, -0.02677539664486202, -0.004155408945657027, -0.004552118824230087, 0.05318923524946798, -0.022276006043768705, -0.011107872874755498, 0.03581435510184399, -0.034488639804044964, -0.027880157530382782, -0.026795483409614594, 0.017927262453029467, 0.006342334503335728, -0.029989247340306547, 0.01789713416854571, 0.0196547074579446, 0.010173847626986485, -0.037119979909600254, -0.009375406178652223, 0.018108041473157498, -0.0019195228302164857, 0.02253713026026192, 0.022276006043768705, -0.005513763373533884, 0.0031586130236975077, -0.011117916257131783, 0.016189774065738047, -0.013006053517422378, -0.04262370129774168, 0.0020475745585302957, -0.01573782744674052, -0.0053380060445939945, 0.024686393599690826, 0.04141850658845807, 0.023722238204792957, -0.02245678506389674, 0.02155288810061148, -0.008486575220253943, -0.05805018389081736, 0.013156703321744104, -0.006944931392316257, 0.023059382418538544, 0.032580415779001795, -0.0012434841732706465, 0.007969346615810561, 0.03400656117527348, 0.043507507770984166, 0.008546835514511653, -0.01704345411714188, 0.0030104745306311275, 0.005719650849602628, -0.030933317833096944, 0.0069499530835044005, 0.01926301927055968, -0.012985966752669808, -0.028201543903778804, 0.023059382418538544, -0.026052280564349897, 0.003989694999093423, -0.017394968775021658, -0.02014682946909237, -0.014110815334265687, -0.004559651361012301, -0.008235493454814466, -0.014783715434218938, 0.02115115839349538, 0.013709083764504484, 0.007939216468681705, -0.021693495453879472, -0.003949521935249558, 0.01832899402279067, -0.015366226024108172, -0.012383370329350553, -0.006513070141087474, 0.031053836558967266, 0.008652290098140097, 0.028643447140400048, 0.024264575265177054, -0.00794925985105799, -0.013136616556991535, -0.020458170597467008, 0.07271338059916597, -0.01957436226157942, 0.024144056539306732, -0.015516874897107348, 0.033584742840759706, -0.03125470048120277, 0.019403626623827674, -0.008210384998873752, 0.009355319413899651, 0.05158230897042317, 0.026012107034844755, -0.023119640850151155, -0.012865448026799486, 0.022215745749510997, -0.0036507341035227265, -0.001078397705274923, -0.02504795163994689, -0.013006053517422378, -0.024626133305433114, 0.013588564107311614, 0.01870059544542302, -0.000321385121931346, 0.03649729765285098, -0.03904829069939599, 0.004572205588982657, -0.02171358221863204, 0.015074970729163554, -0.020769513588486745, -0.026313406643488208, -0.015657480387730238, -0.024867172619818856, -0.03430785892127183, -0.039008117169890846, -0.012594279496607439, 0.031997903326467465, 0.04346733796676922, -0.01202181135777194, -0.028342149394401696, -0.0033293488942798912, -0.0045998244248561665, -0.03938976197489948, -0.022215745749510997, -0.002699132703764578, 0.012162417779717381, 0.03350439950703962, -0.015275836514044155, 0.022557217025014488, -0.03012985469357454, -0.027177128214623227, -0.016460943527252647, 0.021171243295602847, 0.0061364470272669835, 0.002631340571216566, 0.007803632203585682, -0.018730725592551872, 0.0030908208911494955, -0.02410388300980159, -0.019524145349697996, -0.015838259407858272, 0.015145273474475, 0.020397912165854397, 0.010264237137050502, -0.0005172291865199754, -0.02790024429513535, 0.01198163875958935, 0.020417997067961866, -0.026916002135484913, -0.02488725938457143, -0.025771067720459014, -0.02627323311398307, -0.003369521958123757, 0.043507507770984166, 0.007406922325012622, 0.020387868783478112, 0.012423542927533143, 0.014864061561906668, 0.006101295654611261, -0.030973489499956985, -0.03593487196506921, -0.0050341965216790215, 0.002578613279402344, -0.008506661985006513, 0.01000311198923474, -0.04087616766542888, 0.0064779187684317514, -0.015687610534859096, -0.034830111079548455, 0.004336187965785058, -0.013427871851936151, 0.008812982353515558, 0.004632464951917818, -0.014321723570200024, -0.02514838546370974, 0.027980591354145632, -0.02406370948029645, 0.04051460962517281, -0.014150987932448277, -0.00421064708306532, 0.002426708750775495, -0.02669505144849684, 0.005579044893318462, -0.008908393554767715, 0.037260587262868246, 0.031857299698489674, 0.0029652797755991196, 0.014663195777026065, -0.001015627147499735, 0.05029677092741948, -0.012152374397341096, 0.01443219984501661, 0.0006248806436279123, 0.0019308215189744878, 0.04856932405985925, 0.014833931414777813, 0.024746653893948537, 0.016531247203886643, 0.02661470438948656, -0.044391317969516855, -0.026393751839853388, -0.007432030780953334, 0.018298863875661816, 0.06106317252667148, 0.013548391509129022, -0.006934888009939973, -0.024726567129195964, 0.006693849161215505, 0.012222677142652542, -0.04103686178344944, 0.02629331987873564, 0.014673239159402352, 0.017937305835405752, -0.026233059584477927, 0.021793927414997222, 0.012895577242605793, -0.025268904189580062, -0.03872690618864507, 0.0017776614511352841, 0.019855573242825202, -0.02996916243819908, -0.0024254533279784592, 0.005563979819754034, 0.03370526342927513, 0.01777661358003029, -0.02769937851025475, -0.023119640850151155, -0.010344583264738232, -0.016049168575115156, 0.017013323970013025, -0.018379210934672097, 0.044953743657298616, 0.02388293046016842, -0.0012120989525907118, -0.019825444958341445, 0.01606925533986773, 0.029185786063429243, 0.0010388522364142567, 0.01622994759524319, 0.002522119952027653, -0.04346733796676922, 0.01146943091501156, 0.00919462715852419, -0.0126545388595426, 0.021773840650244652, 0.020237218047833835, -0.03822474079512101, 0.014753585287090082, -0.0026840678630307875, 0.007080515657412281, 0.007075493966224138, -0.0019584404712633154, 0.033524482546502, -0.006166576708734564, 0.008918436937144001, -0.008486575220253943, 0.0013658867318284333, -0.00755254997248493, 0.0036080499612541526, 0.01814821500266264, 0.0029602580844109772, 0.01077644405030574, -0.03014994145832711, 0.029607602535297914, -0.011911336014277905, 0.01320691930230298, -0.02635357831034825, 0.009475839071092522, 0.003495063073674133, -0.0024505615510885346, 0.018660423778562977, 0.034629247157312956, 0.011379041404947545, 0.017766570197654007, 0.03215859744448803, -0.048649671118869524, -0.012634452094790031, -0.01868050868067045, -0.04780603444984198, 0.04194075874673469, -0.0030406044449293452, -0.0036858854761784494, -0.02275808280989509, 0.06122386291940184, -0.0452349546385444, 0.050216423868409195, 0.024947519678829137, 0.03844569520739929, 0.009279994977400063, 0.027177128214623227, 0.005493676608781314, 0.0005382572855396848, -0.02137210908048345, 0.0035025956104563463, -0.0014449775531344472, 0.023380766929289466, -0.012554105967102299, -0.01894163475980876, 0.02171358221863204, -0.008064757817062718, -0.004697746471702396, 0.04575720679682102, -0.03509123529604167, -0.03145556812872847, -0.007713242693521666, 0.00660345965115149, 0.026916002135484913, 0.003567876897410287, 0.06411632724145035, -0.005104499266990466, 0.004627443260729676, 0.04676153385857893, 0.03296206058401043, 0.034649333922065526, -0.019303192800064824, -0.006543199822555054, -0.022898688300517982, 0.04668118679956865, -0.03384587078254312, -0.003070734359227563, -0.010505275520113693, -0.014914277542465543, -0.03027046018419743, -0.027036522724000336, -0.0032264051562455195, -0.0031033748862892147, -0.024204316833564443, 0.02647409889886367, -0.024485527814810223, -0.0316363452862114, -0.01987566000757777, -0.010977309835186341, -0.022135398690500716, 0.010214020225169077, -0.001581817357442825, 0.02137210908048345, 0.003495063073674133, 0.021171243295602847, -0.025108211934204597, -0.0006251944993271712, 0.014592893031714621, -0.014110815334265687, 0.023199987909161435, 0.00312848310939929, -0.008657311789328239, -0.031033749794214697, -0.0035703877430043585, -0.03649729765285098, -0.000408636133210432, 0.011379041404947545, 0.006121381953702556, 0.029687949594308195, 0.003841556506027043, 0.009741985910096427, 0.002495756306120542, -0.03189747322799481, 0.012644495477166316, 0.023260248203419143, 0.004070041126781153, 0.00423073384781789, 0.014743541904713798, 0.007637918257022078, -0.008260601910755177, 0.012895577242605793, 0.021994793199877825, 0.022657650848777338, -0.00041114697880450325, -0.040574869919430524, -0.019996180596093194, 0.00535809234368529, -0.01884120279869101, 0.013066313811680089, -0.008004497522805009, 0.004800690209736768, -0.025710809288846403, 0.017977479364910894, -0.012031854740148226, 0.02932639155405213, -0.06383511998549476, 0.03332361862426649, 0.014904234160089258, -0.01597886676112626, -0.022115313788393244, -0.014241377442512294, 0.03179703940423196, 0.04547599209028504, 0.0054133304810935826, -0.029346478318804704, -0.024425267520552515, 0.005940602933574525, -0.0010149994361012171, 0.032540242249496656, 0.0008084844232569332, 0.0073466629620774606, -0.029185786063429243, 0.03434803245077697, -0.012835317879670632, -0.006432724013399743, 0.01886128956344358, -0.023159814379656293, 0.008571943970452365, -0.030390978910067753, 0.02548985673921323, 0.011399128169700115, -0.013578520724935327, 0.010384756794243373, -0.027337820469998688, 0.014874104944282953, 0.005029174830490878, 0.025670635759341264, -0.007100602422164851, 0.01330735219474328, 0.017766570197654007, -0.0005024780850701254, -0.03268084960276465, 0.0059205161688219545, 0.013166746704120389, -0.006020949061262255, 0.007020255828815845, -0.018198431914544063, -0.04684188091758921, 0.0016872718246559496, -0.03752171147936146, 0.015476701367602208, 0.002897487643051108, 0.024405182618445043, 0.006718957617156218, -0.02761903331388957, -0.0343279456860244, -0.009415579708157363, -0.010846747726939736, 0.013658866852623058, 0.020930205843862206, -0.020970379373367345, -0.016591505635499253, -0.019825444958341445, 0.0037561884543205323, -0.025530030268718373, -0.00856190058807608, 0.010088479808110612, 0.02243669829914417, 0.03284153999549501, 0.006593416268775205, 0.006347355728862595, -0.0023099555948043683, -0.02119133006035542, 0.03000933410505912, 0.002663981331108855, -0.027598946549137, -0.005815062050854786, 0.007678091320865943, -0.034669420686818095, -0.00028764595440313985, 0.011559820425075577, 0.0071307321036324315, 0.0035854525837381484, -0.025750982818351545, 0.0006553243554177295, 0.0011656487747616683, 9.658815284842174e-05, -0.011680340082268448, -0.03511132206079424, 0.040072704525906465, -0.010525362284866264, 0.009013848138396159, -0.004978958384270728, -0.00281714128253274, -0.006020949061262255, -0.003507617301644489, 0.028161370374273662, 0.006779216980091379, -0.02886439969003322, -0.043909239340745365, -0.027960504589393063, -0.009028913211960586, 0.011238435914324654, -0.012443629692285714, -0.006799303744843949, 0.024726567129195964, 0.02677539664486202, 0.051060060537436745, -0.00667376286212421, -0.044953743657298616, -0.012112201799158506, -0.002448050705494463, -0.013468044450118741, 0.03023028665469229, 0.009234800688029331, 0.014090728569513118, -0.008225450072438181, 0.018158258385038924, 0.009877570640853726, 0.028101110080015954, 0.0011273587286980376, -0.00812501717999788, 0.02767929360814728, -0.008330904656066623, -0.00951099090940952, 0.020257304812586405, -0.0013018607512562098, -0.025951848603232144, -0.00283722804728531, -0.008903371863579573, -0.03027046018419743, -0.010605708412553995, 0.009822332037784157, -0.004014802989372861, 0.023079469183291113, 0.005764845604634636, -0.017846917256664287, 0.025730896053598972, 0.01522561960216273, 0.01445228660976918, -0.012483803221790855, -0.028161370374273662, 0.007647961173737088, 0.014080685187136831, 0.010917050472251182, -0.016782328038003568, -0.004012292143778789, 0.004976447538676657, 0.010384756794243373, 0.02358163271417007, 0.024686393599690826, -0.0015730295142788943, 0.003575409434192501, 0.04973434523963771, 0.047203438957845276, 0.0031109074230714286, 0.001729955850509205, 0.052626813286976416, -0.0026288297256224944, 0.004075062817969296, -0.015496788132354777, -0.010997396599938912, 0.006397572175082746, 0.008546835514511653, 0.03999236119218639, -0.015155316856851285, 0.03010976792882197, 0.013990295677072816, -0.04190058521722955, 0.007391857717109469, -0.0035954959661144334, -0.0030531584400690643, 0.023159814379656293, 0.0183089072580381, -0.023360680164536896, -0.032379548131476095, 0.0008310818007729372, 0.024144056539306732, -0.00921471392327676, -0.0017864492942992148, -0.017625964707031115, -0.02137210908048345, -0.03179703940423196, 0.010279302210614929, 0.028382322923906834, -0.02773955203975989, 0.014633065629897211, 0.03137522106971819, -0.03535236323782508, -0.029386651848309842, -0.008356013112007336, 0.021914448003512645, -0.017917219070653183, -0.029708036359060764, -0.027257475273633505, -0.010384756794243373, -0.06271026860993124, -0.0055037199911575985, 0.008476532769200207, -0.014843974797154097, -0.01126856513013096, 0.038646559129634786, 0.004994023457835155, 0.015747870829116804, 0.021914448003512645, -0.02516847222846231, -0.03718024020385796, 0.013819560039321069, 0.02358163271417007, 0.03458907362780781, -0.015506831514731064, -0.008451424313259494, -0.03225903126825087, -0.0011091552145563398, -0.03294197381925786, 0.0005552053186766877, -0.008471511078012065, -0.018851246181067295, -0.018580076719552696, 0.01194146523008421, -0.0015893497778097202, 0.002504544032869154, 0.04101677501869686, 0.0064076155574590305, -0.019483971820192854, -0.028161370374273662, 0.045114434050028976, -0.006904758328472392, 0.015627350240601384, -0.007994455071751273, 0.013970208912320245, 0.03029054694895, 0.024545788109067934, 0.009581293654720966, -0.03354456931125457, -0.017726396668148865, -0.00812501717999788, 0.01498458121909954, -0.032017990091220035, 0.049814692298647996, -0.018529859807671273, 0.0034247600955320495, 0.002115366923908945, -0.018529859807671273, -0.020468213979843292, -0.006563286587307624, -0.006739043916247514, -0.007567615046049358, 0.0066436327149953555, 0.025188558993214878, -0.02171358221863204, 0.010515318902489978, -0.03788326951961752, -0.01912241377993679, -0.00268908955421893, -0.004474283542136428, -0.01692293539127156, -0.02263756408402477, -0.015988910143502545, -0.03189747322799481, -0.019805358193588876, -0.015647437005353954, -0.037260587262868246, 0.03151582842298618, 0.004396447794381493, -0.019996180596093194, -0.011901292631901619, 0.007517398599829207, 0.012433586309909428, -0.022095227023640675, 0.013528304744376453, -0.007592723036328795, 0.007150818868385002, 0.027016435959247763, 0.01781678710953543, 0.006432724013399743, -0.01651116043913407, 0.04238266012071084, 0.020417997067961866, -0.01146943091501156, 0.0028246738193149538, 0.017856960639040572, 0.05672447231830853, -0.004160430636845169, 0.0368186821636019, -0.006864585264628527, -0.03171669234522168, 0.043869065811240227, 0.0050819021223051, 0.014442243227392895, 0.006191685164675277, 0.013769343127439644, 0.009782159439601567, 0.006583373352060194, 0.008692462696322687, -0.011891249249525334, -0.00332683804868582, 0.016450900144876362, -0.0164910736743815, 0.05479616152851279, -0.0010275535476562546, 0.01868050868067045, 0.024726567129195964, 0.0164910736743815, -0.0027518599955787993, 0.012644495477166316, 0.018971764906937614, 0.014060598422384262, 0.018037739659168602, 0.02514838546370974, -0.015155316856851285, 0.023722238204792957, 0.04226214325748562, 0.053349929367488544, 0.0196547074579446, -0.02113107162874281, 0.017254363284398767, 0.03458907362780781, -0.033624916370264844, 0.025007778110441747, -0.009199648849712333, -0.008345969729631052, 0.029667862829555625, -0.01789713416854571, -0.011358954640194974, 0.0017575748027980327, -0.0199258769194592, 0.028623362238292576, -0.0006248806436279123, 0.00842129416613064, -0.013940078765191391, 0.020548561038853573, 0.020769513588486745, -0.015275836514044155, -0.02677539664486202, 0.02129176388411827, 0.029507170574180164, 0.019704924369826026, -0.015155316856851285, -0.017244319902022482, -0.01757574779514969, -0.018871331083174764, -0.028241717433283943, -0.020749426823734176, -0.03207825038547774, -0.016792371420379853, -0.001478873619408453, 0.006774195754564511, 0.048810365236890085, 0.04467252895076263, 0.008411250783754353, 0.005262681142433132, -0.013106486409862679, -0.029446910279922453, -0.0214725429042463, 0.005006577220144237, 0.015014710434905845, 0.024385095853692473, -0.04631962875931259, 0.015988910143502545, 0.0017437652684459595, 0.004725365307575905, 0.0008147614790344521, 0.0027970547506108072, 0.011951508612460495, 0.005132118568525251, -0.04656066621105323, 0.030893144303591805, -0.000102159033302608, 0.009606402110661678, 0.009224757305653046, 0.011660253317515877, 0.04551616561979018, 0.014140944550071992, 0.0398517538389184, -0.03440829274503468, 0.038646559129634786, -0.029868728614436228, -0.041619370510693574, -0.007577658428425642, -0.03768240559738202, 0.01563739362297767, 0.00606112212510612, -0.020197046380973794, 0.005202421313836697, -0.011760686209956179, 0.008531770440947226, 0.005835148349946081, -0.00458727019688581, 0.02243669829914417, -0.03611565284784234, -0.01712380117615216, 0.04744447640958591, -0.045636686208305605, -0.0015830727802398606, 0.01057557919674769, -0.016450900144876362, -0.0071106458045411365, 0.03002942086981169, -0.005107010112584538, 0.008883285098827003, -0.027257475273633505, -0.03195773352225252, -0.021593061630116622, -0.05214473465820493, -0.016049168575115156, 0.015968823378749976, 0.0005803135417867629, -0.007979389998186846, 0.023240161438666574, 0.004012292143778789, 0.009596358728285393, -0.053671313878239466, -0.0024819466553531503, 0.003246491688167452, 0.01910232701518422, -0.016481030292005216, -0.01202181135777194, -0.03633660353483042, -0.0069700393825956955, 0.04099668825394429, -0.025449683209708092, -0.03818456726561587, 0.021352024178375978, 0.008958609535326591, 0.023682064675287818, -0.027056609488752905, -0.041980928550949634, 0.00047736989106387986, -0.020458170597467008, 0.010686054540241725, 0.018178345149791494, 0.036075479318337204, -0.02888448645478579, 0.04527512816804954, 0.001107899791759304, 0.020910119079109637, 0.041619370510693574, -0.03400656117527348, -0.043145949730728106, -0.008687441005134544, 0.002694111012576435, -0.0027995655962048784, 0.006076187198670548, 0.04063513021368823, -0.004552118824230087, 0.025268904189580062, 0.016782328038003568, -0.018951678142185045, -0.009099215957272032, -0.00624692330208357, -0.007723286075897951, 0.01728449343152762, 0.016420769997747504, 0.0032590459161378083, 0.006508048449899332, 0.02994907567344651, 0.0352720161788148, -0.05993832487639816, 0.0028698685743469617, 0.013508217979623881, -0.011971595377213064, 0.0017475315368370664, 0.020036352262953232, 0.005885364796166232, 0.043909239340745365, 0.02105072456973253, 0.018178345149791494, 0.033082581172525855, -0.01257419273185487, 0.0657634270500003, 0.019524145349697996, 0.009355319413899651, 0.005543893055001464, -0.04575720679682102, 0.0002761903377955084, 0.027458341058514107, 0.03838543491314157, 0.02119133006035542, 0.036939202752117324, 0.002533418524370336, 0.02050838750934843, -0.014281550972017434, 0.01765609485415997, 0.04483322306878319, -0.016440856762500074, 0.015085014111539839, 0.012704754840101477, -0.035171582355051946, -0.0352720161788148, -0.020387868783478112, 0.027538686254879288, 0.014683282541778636, 0.019132457162313078, -0.023601719478922638, -0.026333493408240777, -0.001323202705975178, -0.020930205843862206, -0.02771946527500732, 0.041257812470437506, -0.03402664794002605, 0.010475146304307388, -0.018379210934672097, -0.012182504544469952, 0.012785101899111756, 0.0003248374764155346, 0.015888476319739695, 0.014361897099705164, -0.008732636225827827, -0.001944631053326561, 0.004486837304445509, -0.024706480364443395, 0.03687894245785961, 0.03004950763456426, 0.006412637248647174, -0.006563286587307624, -0.0033494354262018243, -0.05194386701067923, -0.021110984863990236, 0.014462329992145464, 0.003186232092401654, 0.006452810312491039, 0.06066144095691028, 0.009505969218221378, 0.04945313425839193, -0.03147565489348104, -0.018108041473157498, -0.0035603443606280734, 0.02892465998429093, -0.023702151440040387, 0.021090898099237667, -0.022818343104152802, -0.009485882453468807, 0.003532725291923927, -0.017927262453029467, -0.028583188708787437, -0.017856960639040572, 0.010876876942746042, -0.00345991170101841, -0.009842418802536728, 0.010133674097481345, 0.020417997067961866, 0.04071547727269851, 0.0015730295142788943, -0.01512518670972243, 0.013287265429990711]'' <-> embedding',
        bm25_weight => 0.4,
        similarity_weight => 0.6
    )
    limit 4
)


