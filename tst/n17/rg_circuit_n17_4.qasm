OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
id q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.7898037563179845) q[4];
cx q[1],q[4];
rz(-1.7898037563179845) q[4];
cx q[1],q[4];
rz(2.0847318912809936) q[1];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(3.818419868903417) q[4];
sx q[4];
rz(5.665593818464586) q[4];
sx q[4];
rz(10.150885483293617) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.7565868601084467) q[4];
rz(0.8998222926612245) q[6];
cx q[5],q[6];
rz(-0.8998222926612245) q[6];
cx q[5],q[6];
rz(-pi/2) q[5];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.267375246064428) q[8];
rz(0) q[9];
sx q[9];
rz(9.169571028098936) q[9];
sx q[9];
rz(3*pi) q[9];
rz(pi/2) q[9];
sx q[9];
rz(8.20581059481985) q[9];
sx q[9];
rz(5*pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0) q[10];
sx q[10];
rz(5.485474940770221) q[10];
sx q[10];
rz(3*pi) q[10];
rz(2.056007869301494) q[11];
cx q[8],q[11];
rz(-2.267375246064428) q[11];
sx q[11];
rz(1.7368955190906976) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[8],q[11];
rz(0) q[11];
sx q[11];
rz(4.546289788088888) q[11];
sx q[11];
rz(9.636145337532314) q[11];
rz(pi/4) q[11];
cx q[11],q[4];
rz(-1.7565868601084467) q[4];
cx q[11],q[4];
rz(pi) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[8];
rz(2.125134510290353) q[8];
cx q[6],q[8];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi) q[6];
x q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[9];
rz(2.910158236936501) q[9];
cx q[8],q[9];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[0],q[12];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(-4.600508417471313) q[12];
rz(pi/2) q[12];
cx q[2],q[12];
rz(0) q[12];
sx q[12];
rz(1.5140878937113105) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[2];
sx q[2];
rz(4.769097413468276) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[12];
rz(-pi/2) q[12];
rz(4.600508417471313) q[12];
rz(pi/2) q[12];
sx q[12];
rz(3.60032992203967) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(1.6358190524837175) q[12];
sx q[12];
rz(2.8037449010171964) q[12];
rz(0) q[12];
sx q[12];
rz(3.7497022104099504) q[12];
sx q[12];
rz(3*pi) q[12];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[9],q[12];
rz(0) q[12];
sx q[12];
rz(2.533483096769636) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[9],q[12];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[3],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[3];
cx q[3],q[13];
rz(-pi/4) q[13];
cx q[3],q[13];
cx q[1],q[3];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(-2.0847318912809936) q[3];
cx q[1],q[3];
rz(3.3638166307568143) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[8];
rz(2.0847318912809936) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[8],q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[12],q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[14],q[10];
rz(0) q[10];
sx q[10];
rz(0.7977103664093654) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[14],q[10];
rz(pi/2) q[10];
rz(-2.3549593186693745) q[14];
rz(pi/2) q[14];
cx q[0],q[14];
rz(0) q[0];
sx q[0];
rz(6.141680175472534) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[14];
sx q[14];
rz(0.14150513170705192) q[14];
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
cx q[0],q[13];
rz(2.6698475936014776) q[0];
rz(pi/2) q[13];
rz(-pi/2) q[14];
rz(2.3549593186693745) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[3];
cx q[2],q[13];
cx q[13],q[2];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.184631344510775) q[13];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.2223571188641582) q[3];
cx q[14],q[3];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-3.0889016053877487) q[14];
sx q[14];
rz(4.9722674422462765) q[14];
sx q[14];
rz(12.513679566157128) q[14];
rz(0.6690122002206915) q[14];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[4],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(4.443289293556746) q[2];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-0.31116776435048066) q[4];
sx q[4];
rz(8.080639251835194) q[4];
sx q[4];
rz(9.73594572511986) q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[2],q[4];
rz(1.6799502175866687) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[6],q[0];
rz(-2.6698475936014776) q[0];
cx q[6],q[0];
rz(5.574777869462802) q[0];
rz(-pi/2) q[6];
id q[15];
cx q[15],q[7];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[5],q[7];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(3.280772288869305) q[7];
sx q[7];
rz(3*pi) q[7];
rz(3.7616199118010822) q[7];
cx q[7],q[13];
rz(-3.7616199118010822) q[13];
sx q[13];
rz(2.430080816678444) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[7],q[13];
rz(0) q[13];
sx q[13];
rz(3.8531044905011425) q[13];
sx q[13];
rz(12.001766528059687) q[13];
rz(0) q[7];
sx q[7];
rz(3.205274442698754) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
rz(0) q[7];
sx q[7];
rz(3.0779108644808324) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
cx q[6],q[8];
rz(0) q[6];
sx q[6];
rz(5.46583296737491) q[6];
sx q[6];
rz(3*pi) q[6];
rz(-pi/2) q[6];
rz(3.568478483780903) q[7];
sx q[7];
rz(7.063148472273813) q[7];
cx q[2],q[7];
rz(-1.6799502175866687) q[7];
cx q[2],q[7];
rz(1.0313471982810376) q[2];
sx q[2];
rz(8.715575217994456) q[2];
sx q[2];
rz(11.163371122502518) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.6799502175866687) q[7];
rz(0.5917266225673019) q[7];
sx q[7];
rz(4.139266134299542) q[7];
sx q[7];
rz(8.833051338202077) q[7];
rz(pi/4) q[7];
rz(-pi/2) q[7];
rz(1.1891893565567684) q[8];
sx q[8];
rz(7.536433062206532) q[8];
sx q[8];
rz(8.23558860421261) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-1.5204678866675314) q[16];
sx q[16];
rz(5.285773384962543) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[10];
cx q[10],q[16];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(3.4398438912928464) q[10];
rz(pi/4) q[10];
cx q[10],q[5];
cx q[16],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[16];
cx q[16],q[15];
rz(-pi/4) q[15];
cx q[16],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(4.32176461168109) q[15];
sx q[15];
rz(6.88392215851987) q[15];
sx q[15];
rz(10.900517535835322) q[15];
rz(5.514044490755191) q[15];
sx q[15];
rz(5.913850254893918) q[15];
sx q[15];
rz(12.515213454086606) q[15];
rz(2.897342942444885) q[15];
cx q[13],q[15];
rz(-2.897342942444885) q[15];
cx q[13],q[15];
rz(1.1744199080810347) q[13];
sx q[13];
rz(6.952556958755345) q[13];
sx q[13];
rz(8.250358052688345) q[13];
rz(0.10447393921558959) q[13];
rz(-2.3290169737983897) q[13];
rz(0.6932465145515735) q[15];
rz(pi) q[15];
x q[15];
rz(2.802639975925519) q[15];
sx q[15];
rz(5.262104652557497) q[15];
sx q[15];
rz(14.857694893018758) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[16];
rz(0.459262700667348) q[16];
cx q[0],q[16];
rz(-5.574777869462802) q[16];
sx q[16];
rz(1.9646539246632497) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
cx q[1],q[0];
rz(-pi/4) q[0];
cx q[1],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
rz(6.063771954174408) q[1];
sx q[1];
rz(6.268535427248285) q[1];
sx q[1];
rz(9.6800091547248) q[1];
rz(pi/2) q[1];
cx q[12],q[0];
cx q[0],q[12];
rz(0.26360640111782907) q[0];
sx q[0];
rz(5.382480135818468) q[0];
sx q[0];
rz(10.078367080909231) q[0];
sx q[0];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(3.2735026729445424) q[12];
sx q[12];
rz(5*pi/2) q[12];
x q[12];
rz(0) q[16];
sx q[16];
rz(4.318531382516337) q[16];
sx q[16];
rz(14.540293129564834) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[5];
cx q[10],q[5];
rz(-2.2147970498149148) q[10];
rz(pi/2) q[10];
cx q[11],q[10];
rz(0) q[10];
sx q[10];
rz(1.793405841705807) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[11];
sx q[11];
rz(4.489779465473779) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[11],q[10];
rz(-pi/2) q[10];
rz(2.2147970498149148) q[10];
rz(3.5564701887846475) q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
id q[11];
rz(1.7906787897996757) q[11];
rz(pi/2) q[11];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[14],q[5];
rz(-0.6690122002206915) q[5];
cx q[14],q[5];
rz(pi) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[11];
rz(0) q[11];
sx q[11];
rz(3.07403176349622) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[14];
sx q[14];
rz(3.07403176349622) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[11];
rz(-pi/2) q[11];
rz(-1.7906787897996757) q[11];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
cx q[10],q[14];
x q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(3.7640473187351144) q[14];
cx q[14],q[13];
rz(-3.7640473187351144) q[13];
sx q[13];
rz(2.4341382177519124) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[14],q[13];
cx q[1],q[14];
rz(0) q[13];
sx q[13];
rz(3.849047089427674) q[13];
sx q[13];
rz(15.517842253302884) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[1];
cx q[4],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.8052688229814478) q[10];
rz(pi/4) q[10];
rz(pi/4) q[4];
cx q[4],q[2];
rz(-pi/4) q[2];
cx q[4],q[2];
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
cx q[10],q[2];
rz(-pi/4) q[2];
cx q[10],q[2];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.7485053312339647) q[4];
rz(-0.89673250931508) q[4];
sx q[4];
rz(8.417047485576187) q[4];
sx q[4];
rz(10.321510470084458) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.6690122002206915) q[5];
rz(6.157050790049862) q[5];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[11],q[3];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[3];
cx q[11],q[3];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[13],q[11];
rz(1.5170209197270348) q[11];
cx q[13],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.776396481288153) q[13];
cx q[13],q[14];
rz(-2.776396481288153) q[14];
cx q[13],q[14];
rz(pi/2) q[13];
sx q[13];
rz(4.292664443273617) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
x q[13];
rz(2.776396481288153) q[14];
rz(pi) q[14];
x q[14];
sx q[14];
rz(pi/2) q[3];
cx q[9],q[16];
rz(4.779674621921036) q[16];
cx q[9],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(3.0036483918139796) q[16];
sx q[16];
rz(9.409507012966323) q[16];
sx q[16];
rz(15.16120938700416) q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[16],q[8];
rz(-pi/4) q[8];
cx q[16],q[8];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[3],q[16];
rz(4.228216201239438) q[16];
cx q[3],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(5.702625946391705) q[3];
sx q[3];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-0.23538605632244636) q[8];
sx q[8];
rz(2.7938680704995034) q[8];
rz(0) q[8];
sx q[8];
rz(8.521454776350057) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
x q[9];
cx q[9],q[5];
cx q[5],q[9];
cx q[9],q[5];
rz(pi/2) q[5];
cx q[5],q[0];
rz(pi) q[0];
rz(pi/4) q[0];
cx q[0],q[16];
rz(-pi/4) q[16];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(3.2093551114464125) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[14];
x q[0];
rz(0.6114807044499648) q[0];
sx q[0];
rz(7.325931123890883) q[0];
sx q[0];
rz(8.813297256319414) q[0];
rz(5.906402608865631) q[0];
sx q[0];
rz(7.367600697154752) q[0];
sx q[0];
rz(15.49878436274139) q[0];
rz(pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[14];
x q[14];
rz(2.089526715502469) q[14];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[16];
x q[5];
cx q[5],q[6];
cx q[5],q[12];
cx q[12],q[5];
rz(pi/2) q[12];
cx q[12],q[16];
x q[12];
rz(pi) q[16];
rz(0) q[16];
sx q[16];
rz(5.851020979360134) q[16];
sx q[16];
rz(3*pi) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[6];
cx q[11],q[6];
x q[11];
rz(2.2506820436732475) q[11];
cx q[12],q[11];
rz(2.7800443449615053) q[11];
cx q[12],q[11];
rz(pi/2) q[11];
cx q[11],q[16];
rz(pi/4) q[11];
rz(0.7795961074869506) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[11],q[16];
rz(-pi/4) q[16];
cx q[11],q[16];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(5.4241519203255315) q[16];
sx q[16];
rz(7.780252416624148) q[16];
sx q[16];
rz(10.313345202177244) q[16];
rz(-pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[10];
rz(4.364556215651368) q[10];
rz(-pi/4) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
x q[6];
rz(pi) q[6];
cx q[7],q[5];
rz(pi) q[5];
rz(0.0017907655319557405) q[5];
rz(5.294929225614052) q[5];
cx q[5],q[12];
rz(-5.294929225614052) q[12];
sx q[12];
rz(0.18903332357744373) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[5],q[12];
rz(0) q[12];
sx q[12];
rz(6.094151983602142) q[12];
sx q[12];
rz(13.94011107889648) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[6],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[11],q[6];
rz(4.917357152105413) q[6];
cx q[11],q[6];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
x q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.5912786358322245) q[6];
rz(pi) q[7];
rz(4.442677604638689) q[7];
sx q[7];
rz(5.992875990518008) q[7];
sx q[7];
rz(12.832293032415187) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[9];
cx q[9],q[15];
cx q[1],q[9];
cx q[1],q[8];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(2.944767609678435) q[15];
rz(pi/2) q[15];
cx q[15],q[3];
x q[15];
rz(pi/4) q[15];
cx q[15],q[4];
rz(1.848322972187414) q[3];
rz(pi/2) q[3];
rz(-pi/4) q[4];
cx q[15],q[4];
rz(0.13327954399338132) q[15];
rz(-1.3089130101939146) q[15];
sx q[15];
rz(4.7971037981740094) q[15];
sx q[15];
rz(10.733690970963295) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(5.480340316527027) q[8];
cx q[1],q[8];
rz(2.4012427111216907) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(6.2577213968719825) q[8];
sx q[8];
rz(6.4291205173378945) q[8];
sx q[8];
rz(14.129911230628815) q[8];
rz(pi/4) q[8];
rz(0) q[8];
sx q[8];
rz(5.110854959425688) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[5],q[8];
rz(0) q[8];
sx q[8];
rz(1.1723303477538984) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[5],q[8];
rz(1.5339978480309815) q[5];
rz(4.151945414981631) q[5];
cx q[5],q[14];
rz(-4.151945414981631) q[14];
sx q[14];
rz(0.41153270240560325) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[5],q[14];
rz(0) q[14];
sx q[14];
rz(5.871652604773983) q[14];
sx q[14];
rz(11.487196660248541) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[5];
rz(-pi/2) q[8];
rz(0.04649859946041685) q[9];
cx q[9],q[2];
rz(-0.04649859946041685) q[2];
cx q[9],q[2];
rz(0.04649859946041685) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[3];
rz(0) q[2];
sx q[2];
rz(1.9933430548291793) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[3];
sx q[3];
rz(1.9933430548291793) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[2],q[3];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/4) q[2];
rz(-0.26245307436873383) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(-1.848322972187414) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[4],q[3];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(-0.4428698683837492) q[3];
cx q[4],q[13];
rz(4.165641915197874) q[13];
cx q[4],q[13];
rz(1.250571765497982) q[4];
rz(4.625187960929887) q[4];
rz(pi) q[4];
x q[4];
rz(pi/4) q[4];
cx q[7],q[2];
rz(0) q[2];
sx q[2];
rz(1.9999182620900284) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[7];
sx q[7];
rz(4.283267045089557) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[2];
rz(-pi/2) q[2];
rz(0.26245307436873383) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[15],q[2];
rz(4.608386120523273) q[2];
cx q[15],q[2];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
x q[15];
rz(2.1106689736665536) q[15];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.9391425171800682) q[2];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[12],q[7];
rz(3.5298420172883316) q[7];
cx q[12],q[7];
rz(pi/4) q[12];
cx q[12],q[0];
rz(-pi/4) q[0];
cx q[12],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(-pi/4) q[12];
cx q[0],q[12];
rz(pi/4) q[0];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[14];
rz(1.6331742864273182) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(2.9562848310581606) q[7];
cx q[8],q[2];
rz(-1.9391425171800682) q[2];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
id q[2];
x q[2];
rz(3.357925792055678) q[2];
rz(3.9977628513288743) q[2];
cx q[8],q[6];
rz(-0.5912786358322245) q[6];
cx q[8],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
x q[6];
rz(4.336460934431039) q[9];
sx q[9];
rz(4.60629947865672) q[9];
sx q[9];
rz(10.012253121774117) q[9];
rz(pi/4) q[9];
cx q[9],q[1];
rz(-pi/4) q[1];
cx q[9],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.477238608993899) q[1];
rz(1.0852589159929247) q[1];
cx q[1],q[3];
rz(-1.0852589159929247) q[3];
sx q[3];
rz(1.991836676249792) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(1.36070350856755) q[1];
rz(0) q[3];
sx q[3];
rz(4.291348630929794) q[3];
sx q[3];
rz(10.952906745146054) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.9634105585337838) q[3];
cx q[13],q[3];
rz(-1.9634105585337838) q[3];
cx q[13],q[3];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.453907471069417) q[13];
cx q[16],q[13];
rz(-0.453907471069417) q[13];
cx q[16],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[14];
rz(2.3313749749472588) q[13];
cx q[13],q[8];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[11],q[14];
cx q[14],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0) q[14];
sx q[14];
rz(3.786687775940953) q[14];
sx q[14];
rz(3*pi) q[14];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(-1.36070350856755) q[3];
cx q[1],q[3];
cx q[1],q[15];
rz(-2.1106689736665536) q[15];
cx q[1],q[15];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.36070350856755) q[3];
cx q[3],q[16];
cx q[16],q[12];
cx q[12],q[16];
cx q[16],q[12];
rz(0.2850752263746695) q[12];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[15],q[3];
rz(4.082340083453744) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[11],q[3];
cx q[3],q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[1];
rz(-pi/4) q[1];
cx q[4],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
id q[4];
cx q[6],q[1];
rz(-2.3313749749472588) q[8];
cx q[13],q[8];
rz(-1.328341790117212) q[13];
cx q[2],q[13];
rz(-3.9977628513288743) q[13];
sx q[13];
rz(0.948404362915368) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[2],q[13];
rz(0) q[13];
sx q[13];
rz(5.334780944264218) q[13];
sx q[13];
rz(14.750882602215466) q[13];
rz(2.3313749749472588) q[8];
id q[8];
rz(-2.6069338748350592) q[9];
sx q[9];
rz(3.585473309957142) q[9];
sx q[9];
rz(12.031711835604439) q[9];
rz(5.435778388554994) q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(0) q[10];
sx q[10];
rz(2.7159051982686053) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[9];
sx q[9];
rz(2.7159051982686053) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[10],q[9];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(5.231463045978796) q[10];
sx q[10];
rz(6.946811711604539) q[10];
sx q[10];
rz(15.678229462335187) q[10];
rz(-pi/2) q[9];
rz(-5.435778388554994) q[9];
sx q[9];
cx q[5],q[9];
x q[5];
rz(5.654378963123378) q[5];
sx q[5];
rz(8.846525983710055) q[5];
sx q[5];
rz(12.90135725088945) q[5];
rz(2.8536825105196315) q[9];
cx q[9],q[10];
rz(-2.8536825105196315) q[10];
cx q[9],q[10];
rz(2.8536825105196315) q[10];
cx q[5],q[10];
rz(2.190395569424074) q[10];
cx q[10],q[16];
rz(-2.190395569424074) q[16];
cx q[10],q[16];
rz(2.190395569424074) q[16];
rz(3.070114050137338) q[5];
cx q[5],q[0];
rz(-3.070114050137338) q[0];
cx q[5],q[0];
rz(3.070114050137338) q[0];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[7],q[9];
cx q[12],q[7];
rz(-0.2850752263746695) q[7];
cx q[12],q[7];
rz(0.2850752263746695) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.9446466726152565) q[9];
sx q[9];
rz(6.1818306119662925) q[9];
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
