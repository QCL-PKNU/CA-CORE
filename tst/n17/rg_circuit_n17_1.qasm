OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
x q[0];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[4];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(4.399406040578088) q[4];
rz(pi/2) q[4];
rz(-0.7016053627032338) q[5];
rz(5.135265528800576) q[6];
sx q[6];
rz(3.824479592554524) q[6];
sx q[6];
rz(11.898240347185759) q[6];
rz(3.392710083951065) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(pi/4) q[1];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(0.6090732464865317) q[1];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
cx q[3],q[7];
rz(pi/4) q[3];
rz(pi/2) q[7];
rz(1.4421921527367296) q[8];
rz(0.8538844990275661) q[8];
cx q[8],q[5];
rz(-0.8538844990275661) q[5];
sx q[5];
rz(0.8005187602835235) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[8],q[5];
rz(0) q[5];
sx q[5];
rz(5.482666546896063) q[5];
sx q[5];
rz(10.98026782250018) q[5];
rz(pi/2) q[8];
sx q[8];
rz(7.878473453178636) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[6];
cx q[6],q[8];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.5691098512374639) q[6];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(6.20201045847192) q[9];
sx q[9];
rz(3*pi) q[9];
rz(1.1324118676315613) q[10];
sx q[10];
rz(5.866751934053706) q[10];
sx q[10];
rz(13.012067082882833) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[4];
rz(0) q[10];
sx q[10];
rz(1.0186727171851562) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[4];
sx q[4];
rz(1.0186727171851562) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.25210444380963) q[10];
rz(-pi/2) q[4];
rz(-4.399406040578088) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-0.6782646073931078) q[11];
sx q[11];
rz(8.069667062644609) q[11];
sx q[11];
rz(10.103042568162486) q[11];
rz(2.0296651548421156) q[11];
sx q[11];
rz(7.797607241058772) q[11];
rz(pi/4) q[11];
cx q[11],q[4];
rz(-pi/4) q[4];
cx q[11],q[4];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.1143379669978857) q[4];
rz(2.2451845329927798) q[4];
cx q[4],q[6];
rz(-2.2451845329927798) q[6];
sx q[6];
rz(2.9669701801137114) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[4],q[6];
rz(0) q[6];
sx q[6];
rz(3.316215127065875) q[6];
sx q[6];
rz(11.100852642524695) q[6];
rz(5.356347439757376) q[6];
sx q[6];
rz(8.58799164870127) q[6];
sx q[6];
rz(14.917082358488067) q[6];
rz(pi) q[6];
x q[6];
rz(0) q[6];
sx q[6];
rz(5.639201605828669) q[6];
sx q[6];
rz(3*pi) q[6];
id q[12];
cx q[5],q[12];
cx q[12],q[5];
rz(pi/2) q[12];
cx q[12],q[2];
x q[12];
x q[12];
rz(3.0255637756316762) q[2];
sx q[2];
rz(5.594620364238589) q[2];
rz(pi) q[2];
x q[2];
rz(-pi/2) q[2];
x q[5];
rz(pi/2) q[5];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(3.347958039314382) q[13];
cx q[0],q[13];
cx q[13],q[0];
cx q[0],q[10];
rz(-1.25210444380963) q[10];
cx q[0],q[10];
rz(-pi/4) q[0];
cx q[0],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[12],q[0];
cx q[0],q[12];
x q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.8044241525755336) q[13];
sx q[13];
rz(3.2860890070558395) q[13];
sx q[13];
rz(15.120738408453144) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.9049599535080856) q[14];
sx q[14];
rz(8.977422796936796) q[14];
sx q[14];
rz(11.104142480172488) q[14];
cx q[1],q[14];
rz(-0.6090732464865317) q[14];
cx q[1],q[14];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.6090732464865317) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.4542153767625396) q[14];
cx q[14],q[10];
rz(-1.4542153767625396) q[10];
cx q[14],q[10];
rz(1.4542153767625396) q[10];
rz(3.983563698139309) q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[3],q[1];
rz(-pi/4) q[1];
cx q[3],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(8.052333512616034) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0.5224672952836131) q[1];
sx q[1];
rz(5.978453850137958) q[1];
sx q[1];
rz(8.902310665485766) q[1];
rz(pi) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[3];
cx q[11],q[3];
cx q[3],q[11];
rz(pi) q[11];
x q[11];
rz(pi/4) q[11];
cx q[11],q[5];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(4.261567107719562) q[3];
rz(3.045195795422266) q[3];
cx q[4],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[4];
cx q[4],q[14];
rz(-pi/4) q[14];
cx q[4],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[12],q[14];
rz(3.5978987635950648) q[14];
cx q[12],q[14];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(2.2753963001867743) q[12];
cx q[1],q[12];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.21709975042100838) q[1];
sx q[1];
rz(3.291585833813847) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(6.083663756893721) q[12];
rz(pi) q[12];
x q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(3.8969084394593905) q[14];
rz(pi/2) q[14];
cx q[0],q[14];
rz(0) q[0];
sx q[0];
rz(2.9833085618469597) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[14];
sx q[14];
rz(2.9833085618469597) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[0],q[14];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[14];
rz(-3.8969084394593905) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[4];
rz(-pi/4) q[5];
cx q[11],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi) q[5];
rz(0.15034952488146103) q[5];
sx q[5];
rz(8.152098498423726) q[5];
sx q[5];
rz(12.044245418041559) q[5];
rz(pi/2) q[5];
cx q[15],q[9];
rz(0) q[9];
sx q[9];
rz(0.08117484870766667) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[15],q[9];
rz(pi) q[9];
rz(pi) q[9];
rz(5.185602211086809) q[9];
rz(pi/2) q[9];
sx q[9];
rz(4.245969229544805) q[9];
sx q[9];
rz(5*pi/2) q[9];
cx q[9],q[2];
rz(pi/2) q[2];
cx q[2],q[6];
rz(0) q[6];
sx q[6];
rz(0.6439837013509173) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
cx q[2],q[0];
rz(pi/2) q[0];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(3.047549420215199) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.164200083043369) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[14];
rz(6.097561009586728) q[14];
cx q[9],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-1.9515955398408278) q[14];
sx q[14];
rz(4.758071369322787) q[14];
sx q[14];
rz(11.376373500610207) q[14];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-0.17008324710292677) q[14];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(1.53275212156796) q[9];
sx q[9];
rz(2.5642678438435484) q[9];
rz(pi/4) q[9];
rz(-pi/4) q[16];
rz(0.16718509980809118) q[16];
cx q[16],q[15];
rz(-0.16718509980809118) q[15];
cx q[16],q[15];
rz(0.16718509980809118) q[15];
rz(2.406082147026223) q[15];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[13],q[16];
rz(4.074487073925948) q[16];
cx q[13],q[16];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-1.1073037903935494) q[13];
cx q[10],q[13];
rz(-3.983563698139309) q[13];
sx q[13];
rz(2.212497952921014) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[10],q[13];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[13];
sx q[13];
rz(4.070687354258572) q[13];
sx q[13];
rz(14.515645449302237) q[13];
rz(5.401036628277913) q[13];
sx q[13];
rz(6.388495351019209) q[13];
sx q[13];
rz(14.073555682751351) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.735550733303674) q[16];
rz(pi/2) q[16];
cx q[4],q[10];
rz(-pi/4) q[10];
cx q[4],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.7147002047891179) q[10];
cx q[6],q[10];
rz(-0.7147002047891179) q[10];
cx q[6],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[0],q[10];
cx q[10],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-1.132922232288244) q[6];
sx q[6];
rz(6.149068047405143) q[6];
sx q[6];
rz(10.557700193057624) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.4285941820649808) q[6];
cx q[7],q[15];
rz(-2.406082147026223) q[15];
cx q[7],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(4.405882417681665) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[16];
rz(0) q[16];
sx q[16];
rz(3.1176622719403517) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[7];
sx q[7];
rz(3.1176622719403517) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[16];
rz(-pi/2) q[16];
rz(-2.735550733303674) q[16];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[16];
rz(1.7671896205113613) q[15];
cx q[15],q[8];
rz(3.3709973634390384) q[16];
sx q[16];
rz(7.945003831125944) q[16];
sx q[16];
rz(15.195302209183014) q[16];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[7],q[3];
rz(-3.045195795422266) q[3];
cx q[7],q[3];
cx q[11],q[7];
cx q[3],q[4];
rz(0.3226711191105944) q[3];
sx q[3];
rz(7.834845492065229) q[3];
sx q[3];
rz(14.852983774760034) q[3];
rz(0.3177661848786288) q[4];
sx q[4];
rz(7.766066450461098) q[4];
sx q[4];
rz(11.168849544507436) q[4];
rz(-pi/2) q[4];
rz(2.014350208845978) q[4];
rz(1.6879243396315446) q[7];
cx q[11],q[7];
rz(0.21331181133427513) q[11];
sx q[11];
rz(5.881156330253911) q[11];
sx q[11];
rz(13.303693864453667) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.145449783541863) q[7];
cx q[16],q[7];
rz(-1.145449783541863) q[7];
cx q[16],q[7];
rz(-2.860932174517645) q[16];
rz(pi/2) q[16];
cx q[11],q[16];
rz(0) q[11];
sx q[11];
rz(5.195396147620173) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[16];
sx q[16];
rz(1.0877891595594131) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[11],q[16];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[16];
rz(2.860932174517645) q[16];
cx q[16],q[1];
cx q[1],q[16];
cx q[16],q[1];
cx q[1],q[12];
rz(pi/4) q[1];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(-pi/4) q[12];
cx q[1],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.9430716591111494) q[16];
cx q[16],q[14];
rz(-3.9430716591111494) q[14];
sx q[14];
rz(0.29226138411847646) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[16],q[14];
rz(0) q[14];
sx q[14];
rz(5.99092392306111) q[14];
sx q[14];
rz(13.537932866983455) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(3.0135088569319666) q[16];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
id q[7];
rz(3.428091591978874) q[7];
cx q[7],q[4];
rz(-3.428091591978874) q[4];
sx q[4];
rz(1.150429535712338) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[7],q[4];
rz(0) q[4];
sx q[4];
rz(5.132755771467249) q[4];
sx q[4];
rz(10.838519343902275) q[4];
rz(-1.7671896205113613) q[8];
cx q[15],q[8];
rz(pi/4) q[15];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[15],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
rz(4.347131735223502) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[2];
rz(0.5865236678727987) q[2];
cx q[15],q[2];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[15],q[2];
rz(4.387687764103841) q[15];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0) q[14];
sx q[14];
rz(4.928394846749335) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[14];
sx q[14];
rz(8.284543395813412) q[14];
sx q[14];
rz(3*pi) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[5],q[13];
cx q[13],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[13];
cx q[13],q[11];
rz(-pi/4) q[11];
cx q[13],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
x q[5];
rz(pi/4) q[5];
cx q[5],q[0];
rz(-pi/4) q[0];
cx q[5],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.057863021993512) q[0];
sx q[0];
rz(6.364961121275796) q[0];
sx q[0];
rz(14.541328755245118) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(4.828567019321318) q[5];
rz(pi/2) q[5];
rz(1.7671896205113613) q[8];
rz(1.1706899467527425) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(2.7919658016615125) q[3];
sx q[3];
rz(3.8128536271222457) q[3];
sx q[3];
rz(11.016116382036492) q[3];
rz(1.9710171091895838) q[3];
cx q[3],q[11];
rz(-1.9710171091895838) q[11];
cx q[3],q[11];
rz(1.9710171091895838) q[11];
cx q[11],q[12];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[6];
rz(-0.4285941820649808) q[6];
cx q[8],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.098768445597639) q[6];
cx q[4],q[6];
rz(-3.098768445597639) q[6];
cx q[4],q[6];
cx q[4],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.5220065756272656) q[0];
rz(-2.3801947365912794) q[4];
cx q[0],q[4];
rz(-2.5220065756272656) q[4];
sx q[4];
rz(0.6501023884172454) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[0],q[4];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[4];
sx q[4];
rz(5.633082918762341) q[4];
sx q[4];
rz(14.326979272987924) q[4];
rz(-0.28504154616475774) q[4];
sx q[4];
rz(6.331105751474204) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.2998777375660304) q[6];
cx q[16],q[6];
rz(-3.0135088569319666) q[6];
sx q[6];
rz(2.067878356684032) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[16],q[6];
rz(0) q[6];
sx q[6];
rz(4.215306950495554) q[6];
sx q[6];
rz(11.138409080135315) q[6];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[5];
rz(0) q[5];
sx q[5];
rz(2.478428140247192) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[8];
sx q[8];
rz(2.478428140247192) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[5];
rz(-pi/2) q[5];
rz(-4.828567019321318) q[5];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[1],q[8];
rz(4.977394437792797) q[1];
rz(6.023259012019546) q[1];
sx q[1];
rz(4.157693279152209) q[1];
sx q[1];
rz(9.998970362324947) q[1];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[8];
cx q[9],q[10];
rz(-pi/4) q[10];
cx q[9],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
cx q[13],q[10];
rz(pi/2) q[10];
cx q[10],q[3];
rz(pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[13];
sx q[13];
rz(1.93948654799182) q[13];
sx q[13];
rz(4.6760583482895814) q[13];
sx q[13];
rz(12.283218115876489) q[13];
x q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.8198880516983756) q[3];
cx q[10],q[3];
id q[10];
cx q[10],q[2];
rz(0) q[10];
sx q[10];
rz(3.471918728824425) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.4693910689273233) q[2];
rz(-5.170924565353838) q[2];
rz(pi/2) q[2];
cx q[3],q[16];
cx q[16],q[3];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[3],q[16];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
cx q[1],q[16];
rz(1.3323034448250461) q[16];
cx q[1],q[16];
rz(2.988094313196087) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[14];
rz(5.66232887281742) q[14];
cx q[3],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(2.1901071760516015) q[14];
sx q[14];
rz(4.575375303305851) q[14];
sx q[14];
rz(14.502050473419695) q[14];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[16];
rz(0) q[16];
sx q[16];
rz(0.48995346183140187) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[3];
sx q[3];
rz(0.48995346183140187) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[16];
rz(-pi/2) q[16];
rz(-2.988094313196087) q[16];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[9],q[7];
rz(3.4415642109222087) q[7];
cx q[9],q[7];
rz(-1.082431256694817) q[7];
cx q[15],q[7];
rz(-4.387687764103841) q[7];
sx q[7];
rz(0.5987726687994375) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(2.1480964853017626) q[15];
sx q[15];
rz(4.040238571016504) q[15];
sx q[15];
rz(10.951117132166452) q[15];
rz(1.9278066587054683) q[15];
sx q[15];
rz(6.020045327740371) q[15];
sx q[15];
rz(14.79385949116789) q[15];
rz(-pi/2) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[2];
rz(0) q[15];
sx q[15];
rz(4.719612732955811) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[2];
sx q[2];
rz(1.5635725742237745) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[15],q[2];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(1.839073099888267) q[15];
rz(0) q[15];
sx q[15];
rz(4.009788466876586) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[2];
rz(5.170924565353838) q[2];
rz(0) q[7];
sx q[7];
rz(5.684412638380149) q[7];
sx q[7];
rz(14.894896981568039) q[7];
rz(pi) q[7];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[7];
rz(-pi/4) q[7];
cx q[8],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cx q[7],q[0];
rz(-pi/4) q[0];
cx q[7],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(5.771254530495257) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
rz(0) q[9];
sx q[9];
rz(0.5119307766843293) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
cx q[5],q[6];
cx q[6],q[5];
id q[5];
rz(pi/2) q[5];
cx q[8],q[5];
cx q[5],q[8];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(5.027117477035387) q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[5];
rz(0.32210496850691805) q[2];
rz(pi/2) q[2];
rz(2.0648928020145165) q[8];
sx q[8];
rz(5.481958861628751) q[8];
rz(0.22300918968033226) q[8];
rz(pi/4) q[9];
cx q[9],q[11];
rz(-pi/4) q[11];
cx q[9],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[6];
cx q[6],q[11];
rz(0.6307547536166833) q[11];
sx q[11];
rz(4.556764752686114) q[11];
sx q[11];
rz(8.794023207152696) q[11];
cx q[11],q[1];
cx q[1],q[11];
cx q[11],q[1];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[3];
cx q[14],q[1];
rz(4.930623801168542) q[1];
cx q[14],q[1];
rz(-2.1834206152888984) q[1];
id q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[14];
rz(0.5073271906156963) q[14];
cx q[15],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(3.671239908727832) q[3];
cx q[11],q[3];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
sx q[11];
cx q[2],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.5652270766320777) q[11];
x q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.749443664967052) q[3];
sx q[6];
cx q[6],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[6];
cx q[6],q[0];
rz(-pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[16];
cx q[16],q[0];
rz(-1.6367748405759395) q[16];
sx q[16];
rz(6.5544317363642755) q[16];
sx q[16];
rz(11.061552801345318) q[16];
rz(2.7420318260234637) q[16];
sx q[16];
rz(6.610027042217034) q[16];
cx q[16],q[14];
rz(6.126704410778635) q[14];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[3],q[0];
rz(-2.749443664967052) q[0];
cx q[3],q[0];
rz(2.749443664967052) q[0];
rz(-pi/4) q[0];
cx q[0],q[11];
rz(-2.5652270766320777) q[11];
cx q[0],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[6],q[8];
rz(-0.22300918968033226) q[8];
cx q[6],q[8];
rz(2.1948164168211313) q[6];
rz(pi/2) q[6];
cx q[13],q[6];
rz(0) q[13];
sx q[13];
rz(0.9510379766960209) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[6];
sx q[6];
rz(0.9510379766960209) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[13],q[6];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(-pi/2) q[6];
rz(-2.1948164168211313) q[6];
rz(5.158456619166454) q[6];
cx q[6],q[15];
rz(pi) q[8];
x q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-2.6530963152407003) q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
cx q[9],q[12];
rz(-pi/4) q[12];
cx q[9],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[10];
rz(0) q[10];
sx q[10];
rz(2.8112665783551614) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[12],q[10];
rz(0) q[10];
sx q[10];
rz(4.7523517196524345) q[10];
sx q[10];
rz(3*pi) q[10];
x q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[4],q[9];
cx q[4],q[10];
rz(0) q[10];
sx q[10];
rz(1.5308335875271513) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[4],q[10];
rz(pi/2) q[10];
rz(-3.8807558782860827) q[4];
sx q[4];
rz(5.359505837294965) q[4];
sx q[4];
rz(13.305533839055462) q[4];
cx q[7],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(2.5173030486993313) q[12];
cx q[12],q[1];
rz(-2.5173030486993313) q[1];
sx q[1];
rz(1.780617883907745) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[12],q[1];
rz(0) q[1];
sx q[1];
rz(4.502567423271842) q[1];
sx q[1];
rz(14.125501624757609) q[1];
cx q[1],q[13];
rz(2.230185740276261) q[1];
rz(pi/2) q[1];
rz(0.8043545712831719) q[12];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(5.406056264344042) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[10];
cx q[10],q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[4],q[10];
cx q[10],q[4];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[12],q[4];
rz(-0.8043545712831719) q[4];
cx q[12],q[4];
rz(pi) q[12];
x q[12];
rz(0.8043545712831719) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[8];
rz(0) q[4];
sx q[4];
rz(3.367105098828537) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[7],q[10];
rz(-pi/4) q[10];
cx q[7],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[2],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[2];
cx q[2],q[10];
rz(-pi/4) q[10];
cx q[2],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[7];
sx q[7];
rz(4.33799444857622) q[7];
sx q[7];
rz(3*pi) q[7];
rz(0) q[8];
sx q[8];
rz(2.9160802083510493) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[4],q[8];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[8];
rz(2.6530963152407003) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[5],q[9];
rz(0.4848748393297202) q[9];
cx q[5],q[9];
rz(pi/4) q[5];
cx q[5],q[3];
rz(-pi/4) q[3];
cx q[5],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.9392678107621584) q[3];
cx q[3],q[13];
rz(-1.9392678107621584) q[13];
cx q[3],q[13];
rz(1.9392678107621584) q[13];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[1];
rz(0) q[1];
sx q[1];
rz(2.3010560761733165) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[5];
sx q[5];
rz(2.3010560761733165) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[1];
rz(-pi/2) q[1];
rz(-2.230185740276261) q[1];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];