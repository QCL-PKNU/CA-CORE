OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(0) q[3];
sx q[3];
rz(6.261766479092825) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[5],q[4];
rz(1.1084505910476008) q[4];
cx q[5],q[4];
rz(pi/2) q[4];
sx q[4];
rz(4.012886101243234) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(2.798856101109532) q[4];
sx q[4];
rz(7.466559573430794) q[4];
sx q[4];
rz(9.595918712641204) q[4];
rz(pi) q[4];
rz(pi/2) q[4];
id q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[3];
rz(0) q[3];
sx q[3];
rz(0.021418828086761543) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[8],q[3];
rz(2.725198549379857) q[3];
cx q[5],q[3];
rz(-2.725198549379857) q[3];
cx q[5],q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(2.328652931362954) q[3];
sx q[3];
cx q[4],q[3];
rz(1.783065091095301) q[3];
x q[4];
sx q[5];
rz(pi/4) q[8];
rz(2.6661381206667936) q[9];
rz(5.936065061405347) q[9];
rz(2.6153508503481318) q[10];
cx q[10],q[2];
rz(-2.6153508503481318) q[2];
cx q[10],q[2];
sx q[10];
rz(2.6153508503481318) q[2];
rz(pi/2) q[2];
cx q[2],q[10];
cx q[10],q[8];
x q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(6.1138023351694555) q[8];
cx q[10],q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[11],q[7];
cx q[7],q[11];
cx q[11],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
id q[12];
rz(pi) q[12];
x q[12];
rz(3.280148417737854) q[13];
sx q[13];
rz(4.720675215839149) q[13];
sx q[13];
rz(15.308219262202211) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi) q[13];
x q[13];
rz(-pi/2) q[13];
sx q[13];
cx q[1],q[14];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
cx q[11],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
cx q[1],q[2];
rz(pi/4) q[1];
rz(0) q[11];
sx q[11];
rz(8.355659064957056) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[14];
cx q[14],q[6];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
rz(5.166636939553983) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(-pi/4) q[6];
cx q[14],q[6];
cx q[12],q[14];
rz(5.067786198095903) q[14];
cx q[12],q[14];
rz(pi) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[5];
rz(1.2635058567775177) q[5];
cx q[5],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(2.547005346161326) q[2];
cx q[2],q[3];
rz(-2.547005346161326) q[3];
sx q[3];
rz(2.69601688619733) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[2],q[3];
rz(6.0967154600505316) q[2];
sx q[2];
rz(8.859229423407697) q[2];
sx q[2];
rz(13.257420667937689) q[2];
rz(0.4948911870426144) q[2];
rz(0) q[3];
sx q[3];
rz(3.5871684209822563) q[3];
sx q[3];
rz(10.188718215835404) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
x q[6];
rz(-3.934117026434441) q[6];
sx q[6];
rz(3.7107818309358302) q[6];
sx q[6];
rz(13.35889498720382) q[6];
rz(3.7712282740462992) q[15];
rz(pi/4) q[15];
cx q[16],q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
cx q[0],q[7];
rz(pi/4) q[0];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[15],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(pi/4) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(-pi/4) q[16];
cx q[15],q[16];
rz(pi) q[15];
x q[15];
rz(1.2682606876881104) q[15];
rz(1.7026153327714646) q[15];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[0],q[7];
rz(-pi/4) q[7];
cx q[0],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[14];
rz(4.617021991684291) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[1];
cx q[1],q[14];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(1.51567245161598) q[14];
rz(2.1458155284304294) q[14];
cx q[5],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(6.223404215981177) q[5];
sx q[5];
rz(7.188677647626759) q[5];
sx q[5];
rz(15.424950019452334) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[7],q[6];
rz(3.551109097250143) q[6];
cx q[7],q[6];
rz(pi/4) q[6];
rz(pi/2) q[7];
cx q[8],q[16];
rz(2.0461030605339907) q[16];
cx q[8],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[11],q[16];
cx q[16],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.5101685174890935) q[16];
rz(-1.4113252068201783) q[16];
rz(pi/2) q[16];
cx q[4],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.9911977574090034) q[11];
rz(4.386374800790619) q[4];
rz(1.43476097365983) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[15],q[8];
rz(-1.7026153327714646) q[8];
cx q[15],q[8];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(1.7026153327714646) q[8];
cx q[9],q[0];
rz(2.011486032627144) q[0];
cx q[0],q[10];
rz(-2.011486032627144) q[10];
cx q[0],q[10];
rz(pi/2) q[0];
cx q[0],q[13];
x q[0];
rz(2.011486032627144) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[0],q[10];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[3];
rz(pi/4) q[10];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[10],q[3];
rz(-pi/4) q[3];
cx q[10],q[3];
rz(pi/4) q[10];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(-0.4948911870426144) q[3];
cx q[2],q[3];
rz(0.3298508740847689) q[2];
sx q[2];
rz(5.740218288653173) q[2];
sx q[2];
rz(10.3042146315448) q[2];
rz(0.4948911870426144) q[3];
cx q[6],q[13];
rz(-pi/4) q[13];
cx q[6],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[16];
rz(0) q[13];
sx q[13];
rz(5.4700200228467315) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[16];
sx q[16];
rz(0.8131652843328547) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[13],q[16];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[16];
rz(1.4113252068201783) q[16];
rz(0.1505858500477063) q[16];
rz(3.5881008005562967) q[16];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[8],q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[13],q[8];
rz(5.468462020481098) q[8];
cx q[13],q[8];
sx q[13];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.22112763922307208) q[9];
sx q[9];
cx q[12],q[9];
x q[12];
rz(-2.0291601663430963) q[12];
cx q[14],q[12];
rz(-2.1458155284304294) q[12];
sx q[12];
rz(0.9976522010846551) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[14],q[12];
rz(0) q[12];
sx q[12];
rz(5.285533106094931) q[12];
sx q[12];
rz(13.599753655542905) q[12];
cx q[12],q[1];
rz(2.5571341480775467) q[1];
cx q[12],q[1];
rz(3.1270019447790465) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[12],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[12];
cx q[12],q[0];
rz(-pi/4) q[0];
cx q[12],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-0.8144920082605944) q[14];
cx q[4],q[14];
rz(-1.43476097365983) q[14];
sx q[14];
rz(2.368890928410642) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[4],q[14];
rz(0) q[14];
sx q[14];
rz(3.914294378768944) q[14];
sx q[14];
rz(11.674030942689804) q[14];
rz(-pi/4) q[14];
rz(-pi/4) q[14];
rz(1.1686910958235248) q[4];
cx q[16],q[4];
rz(-3.5881008005562967) q[4];
sx q[4];
rz(3.0733801076954403) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[16],q[4];
cx q[16],q[3];
cx q[3],q[16];
rz(pi/4) q[16];
rz(0) q[4];
sx q[4];
rz(3.209805199484146) q[4];
sx q[4];
rz(11.844187665502151) q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[9];
cx q[7],q[9];
x q[7];
rz(1.215528671805418) q[7];
cx q[7],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
cx q[15],q[13];
rz(pi/2) q[13];
x q[15];
cx q[15],q[2];
cx q[2],q[15];
cx q[15],q[2];
rz(5.506513712019005) q[2];
rz(2.329626146001531) q[2];
rz(2.6468489740778023) q[7];
rz(3.5964084053821352) q[7];
sx q[7];
rz(2.8132601180412005) q[7];
rz(1.5322476846307795) q[7];
cx q[9],q[11];
rz(-1.9911977574090034) q[11];
cx q[9],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[6],q[11];
rz(0.7442211292160809) q[11];
cx q[6],q[11];
cx q[0],q[6];
cx q[11],q[12];
rz(pi/4) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[11],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
rz(0.4914224517813273) q[11];
sx q[11];
rz(3.8898754990578914) q[11];
sx q[11];
rz(14.018274360528684) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(0.6117135681499983) q[12];
cx q[1],q[12];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/4) q[1];
rz(0.059415882941198817) q[1];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(5.230178156335467) q[6];
cx q[0],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[16],q[0];
rz(-pi/4) q[0];
cx q[16],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.3472762554522641) q[0];
sx q[0];
rz(7.083034288427812) q[0];
sx q[0];
rz(10.772054216221644) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.9920251617595117) q[16];
cx q[2],q[16];
rz(-2.329626146001531) q[16];
sx q[16];
rz(0.7686605318663187) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[2],q[16];
rz(0) q[16];
sx q[16];
rz(5.514524775313268) q[16];
sx q[16];
rz(13.746429268530422) q[16];
rz(pi) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.005328865410889786) q[2];
cx q[11],q[2];
rz(-0.005328865410889786) q[2];
cx q[11],q[2];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[14],q[6];
rz(pi/4) q[14];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[14],q[6];
rz(-pi/4) q[6];
cx q[14],q[6];
rz(0.29776430968770445) q[14];
rz(4.609976679819287) q[14];
rz(pi/4) q[14];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.2855642288122672) q[6];
cx q[6],q[13];
rz(-0.2855642288122672) q[13];
cx q[6],q[13];
rz(0.2855642288122672) q[13];
rz(2.772311835119489) q[13];
cx q[13],q[1];
rz(-2.772311835119489) q[1];
sx q[1];
rz(0.21217490289604646) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[13],q[1];
rz(0) q[1];
sx q[1];
rz(6.071010404283539) q[1];
sx q[1];
rz(12.137673912947669) q[1];
rz(pi/2) q[1];
rz(pi/4) q[13];
rz(3.026494308510107) q[13];
rz(pi/2) q[13];
cx q[6],q[0];
rz(2.7187144090419078) q[0];
cx q[6],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.2776602562091037) q[0];
sx q[0];
rz(7.575194321533233) q[0];
rz(0) q[0];
sx q[0];
rz(3.924807176318749) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/2) q[0];
sx q[6];
cx q[1],q[6];
x q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(2.447628408975521) q[6];
x q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[7],q[16];
cx q[16],q[7];
cx q[7],q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(-pi/4) q[9];
cx q[10],q[9];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[8];
rz(5.52345845492936) q[8];
cx q[10],q[8];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[5];
rz(1.2816098875643784) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.7032057702349119) q[8];
cx q[3],q[8];
rz(-0.7032057702349119) q[8];
cx q[3],q[8];
rz(-pi/4) q[3];
rz(1.9365612166693138) q[3];
sx q[3];
rz(7.600704793374802) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[15],q[8];
rz(5.984768060658842) q[8];
cx q[15],q[8];
rz(pi) q[15];
x q[15];
rz(2.8033162850354785) q[15];
rz(pi/2) q[15];
cx q[16],q[15];
rz(0) q[15];
sx q[15];
rz(0.17365092726449438) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[16];
sx q[16];
rz(0.17365092726449438) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[16],q[15];
rz(-pi/2) q[15];
rz(-2.8033162850354785) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
id q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[14],q[8];
rz(-pi/4) q[8];
cx q[14],q[8];
rz(0.04977911302171452) q[14];
sx q[14];
rz(6.711390656199551) q[14];
sx q[14];
rz(13.041851561482641) q[14];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(3.8366248637299933) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.812632752143716) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[4];
rz(1.21232648433532) q[4];
cx q[9],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(-pi/4) q[4];
cx q[12],q[4];
rz(2.820130261414861) q[12];
cx q[3],q[12];
cx q[12],q[3];
cx q[3],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[12];
rz(5.169623657017442) q[12];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(pi/4) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
rz(5.490172539607816) q[0];
rz(-0.8488946617571401) q[0];
sx q[0];
rz(5.104064270570271) q[0];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.1207405634783503) q[12];
rz(pi) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[2],q[3];
rz(pi/4) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(-pi/4) q[3];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(4.562227805864453) q[2];
sx q[2];
rz(2.449666076599347) q[2];
rz(4.26453835230859) q[2];
sx q[2];
rz(5.362030615172701) q[2];
sx q[2];
rz(13.139590374723301) q[2];
rz(0.6304985681649777) q[2];
sx q[2];
rz(3.429727816544537) q[2];
sx q[2];
rz(10.898634727511618) q[2];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[3];
id q[3];
rz(0.7263232612184968) q[3];
sx q[3];
rz(7.203735135831468) q[3];
sx q[3];
rz(14.914075802149242) q[3];
rz(pi/2) q[3];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[5];
rz(0.2906187965803249) q[4];
rz(pi/2) q[5];
rz(5.423533484512603) q[5];
sx q[5];
rz(7.9709800231998456) q[5];
sx q[5];
rz(12.339725226710822) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.28276205965795664) q[5];
cx q[7],q[4];
rz(-0.2906187965803249) q[4];
cx q[7],q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[15],q[4];
rz(1.8440640458906425) q[4];
cx q[15],q[4];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(5.084087377705407) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(4.345619355635101) q[15];
sx q[15];
rz(4.912300574869809) q[15];
rz(0.7269748188836913) q[15];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
id q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[7],q[5];
rz(-0.28276205965795664) q[5];
cx q[7],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.7903527758886266) q[5];
sx q[5];
rz(3.7257011416641816) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(2.0721256949591207) q[7];
cx q[7],q[1];
rz(-2.0721256949591207) q[1];
cx q[7],q[1];
rz(2.0721256949591207) q[1];
rz(pi/2) q[1];
sx q[1];
rz(5.827977888559423) q[1];
sx q[1];
rz(5*pi/2) q[1];
sx q[1];
cx q[7],q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(0.5700254182154114) q[5];
sx q[5];
rz(4.252395787811445) q[5];
sx q[5];
rz(14.996136237932458) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[14];
rz(5.484896964280501) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.7363754775271727) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[9];
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
cx q[9],q[10];
rz(1.1485647457562649) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[13];
rz(0) q[10];
sx q[10];
rz(2.3946090831352236) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[13];
sx q[13];
rz(2.3946090831352236) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[10],q[13];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
cx q[10],q[11];
rz(3.5879645956051966) q[11];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
rz(-pi/4) q[11];
rz(-pi/2) q[13];
rz(-3.026494308510107) q[13];
cx q[13],q[8];
rz(-1.812632752143716) q[8];
cx q[13],q[8];
rz(-0.82154742725429) q[13];
rz(pi/2) q[13];
cx q[16],q[13];
rz(0) q[13];
sx q[13];
rz(0.7350482231146396) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[16];
sx q[16];
rz(5.548137084064947) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[16],q[13];
rz(-pi/2) q[13];
rz(0.82154742725429) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(0.37064726651152724) q[13];
sx q[13];
rz(7.539506642167601) q[13];
sx q[13];
rz(9.054130694257852) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[0],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(1.0227018259001177) q[16];
cx q[12],q[16];
rz(-2.1207405634783503) q[16];
sx q[16];
rz(0.34156826395671125) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[12],q[16];
rz(0) q[12];
sx q[12];
rz(5.657577279398882) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[16];
sx q[16];
rz(5.941617043222875) q[16];
sx q[16];
rz(10.522816698347611) q[16];
rz(1.356988064414961) q[16];
rz(4.6558461455950155) q[16];
cx q[16],q[15];
rz(-4.6558461455950155) q[15];
sx q[15];
rz(0.9174874921500695) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[16],q[15];
rz(0) q[15];
sx q[15];
rz(5.365697815029517) q[15];
sx q[15];
rz(13.353649287480703) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-0.39906707826754784) q[16];
rz(pi/2) q[16];
cx q[15],q[16];
rz(0) q[15];
sx q[15];
rz(5.027808580299083) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[16];
sx q[16];
rz(1.2553767268805034) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[15],q[16];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.5295966601405072) q[15];
cx q[13],q[15];
rz(-0.5295966601405072) q[15];
cx q[13],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[16];
rz(0.39906707826754784) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[5],q[12];
rz(0) q[12];
sx q[12];
rz(0.6256080277807046) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[5],q[12];
rz(2.0080187971288397) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(8.544944899212545) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[10],q[8];
rz(5.7867060730671405) q[8];
cx q[10],q[8];
rz(pi) q[10];
x q[10];
cx q[10],q[5];
rz(-2.0080187971288397) q[5];
cx q[10],q[5];
rz(3.1308808073606933) q[5];
cx q[0],q[5];
rz(-3.1308808073606933) q[5];
cx q[0],q[5];
rz(0.251720474447488) q[9];
sx q[9];
rz(4.286025439267318) q[9];
sx q[9];
rz(12.4432357856729) q[9];
rz(0.7261758802660796) q[9];
rz(pi/2) q[9];
cx q[9],q[4];
rz(1.870438849312412) q[4];
cx q[9],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[1];
cx q[11],q[1];
rz(0) q[11];
sx q[11];
rz(3.6459215091804036) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[14],q[1];
sx q[1];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[3],q[1];
x q[3];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[4];
rz(1.4926863424446015) q[4];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[12],q[4];
rz(pi) q[12];
x q[12];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi) q[4];
x q[4];
cx q[8],q[11];
rz(0) q[11];
sx q[11];
rz(2.6372637979991826) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[8],q[11];
rz(pi/4) q[11];
cx q[11],q[14];
rz(-pi/4) q[14];
cx q[11],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[8];
rz(pi) q[9];
x q[9];
rz(-pi/2) q[9];
cx q[9],q[7];
rz(0.8887570665639883) q[7];
cx q[9],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.5522930184312808) q[7];
cx q[7],q[10];
rz(-0.5522930184312808) q[10];
cx q[7],q[10];
rz(0.5522930184312808) q[10];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
rz(pi/4) q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
rz(-pi/4) q[9];
cx q[6],q[9];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
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