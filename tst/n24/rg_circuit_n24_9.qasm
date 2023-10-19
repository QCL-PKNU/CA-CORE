OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.4007509080695733) q[2];
sx q[2];
rz(4.205792441272857) q[2];
sx q[2];
rz(14.16794482901386) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[4];
rz(-3.5859442780846402) q[4];
sx q[4];
rz(3.6360905039003404) q[4];
sx q[4];
rz(13.010722238854019) q[4];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi) q[6];
rz(1.6004763235957138) q[7];
sx q[7];
rz(3.736252179826804) q[7];
rz(1.015376083272757) q[7];
sx q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.3414755673575067) q[9];
rz(pi) q[9];
x q[9];
rz(0) q[9];
sx q[9];
rz(4.76588630472339) q[9];
sx q[9];
rz(3*pi) q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[12];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[1];
rz(4.708535149823544) q[1];
cx q[14],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[8];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.6811353124850785) q[14];
rz(3.729561158035354) q[8];
cx q[1],q[8];
rz(-5.3982644773558786) q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[10];
rz(2.4163983444012893) q[10];
cx q[15],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.5771380452763162) q[10];
cx q[10],q[12];
rz(-1.5771380452763162) q[12];
cx q[10],q[12];
rz(-pi/2) q[10];
rz(1.5771380452763162) q[12];
rz(4.042363258776854) q[12];
sx q[12];
rz(7.6390785336282345) q[12];
sx q[12];
rz(9.80333622570829) q[12];
rz(-3.667081780895316) q[12];
sx q[12];
rz(8.757075575164224) q[12];
sx q[12];
rz(13.091859741664695) q[12];
rz(pi/4) q[12];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.8173617721626325) q[15];
cx q[15],q[7];
rz(-1.8173617721626325) q[7];
sx q[7];
rz(2.1355249311951585) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
rz(0) q[7];
sx q[7];
rz(4.147660375984428) q[7];
sx q[7];
rz(10.226763649659254) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi) q[16];
x q[16];
rz(2.4458017272540338) q[16];
cx q[16],q[8];
rz(-pi/2) q[16];
rz(0.9790915204055922) q[8];
sx q[8];
rz(4.221460993250896) q[8];
sx q[8];
rz(9.487203229202075) q[8];
rz(-pi/2) q[8];
x q[17];
rz(0) q[17];
sx q[17];
rz(4.92149425791437) q[17];
sx q[17];
rz(3*pi) q[17];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[19];
cx q[19],q[11];
rz(-pi/4) q[11];
cx q[19],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-3.6665857067100913) q[11];
rz(pi/2) q[11];
rz(-pi/4) q[19];
rz(0.1080846204531013) q[19];
rz(5.651030246111154) q[20];
rz(pi/2) q[20];
cx q[3],q[20];
rz(0) q[20];
sx q[20];
rz(1.5082529640626217) q[20];
sx q[20];
rz(3*pi) q[20];
rz(0) q[3];
sx q[3];
rz(1.5082529640626217) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[20];
rz(-pi/2) q[20];
rz(-5.651030246111154) q[20];
cx q[20],q[14];
rz(-2.6811353124850785) q[14];
cx q[20],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(1.1724551087476076) q[3];
sx q[3];
rz(4.277567752611399) q[3];
sx q[3];
rz(12.230188039349237) q[3];
rz(pi/4) q[3];
cx q[4],q[20];
cx q[20],q[4];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(2.051310133083238) q[21];
cx q[21],q[13];
rz(-2.051310133083238) q[13];
cx q[21],q[13];
rz(2.051310133083238) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[21];
cx q[5],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[5];
cx q[5],q[13];
rz(-pi/4) q[13];
cx q[5],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[17],q[13];
rz(5.929326715627042) q[13];
cx q[17],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[4],q[13];
cx q[13],q[4];
rz(4.285801615044509) q[13];
rz(1.3802044537716305) q[4];
rz(0) q[5];
sx q[5];
rz(3.2499431099619467) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[15],q[5];
rz(0) q[5];
sx q[5];
rz(3.0332421972176395) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[15],q[5];
cx q[19],q[15];
rz(-0.1080846204531013) q[15];
cx q[19],q[15];
rz(0.1080846204531013) q[15];
rz(pi/2) q[15];
sx q[15];
rz(3.6182886270020553) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(4.575136095484395) q[19];
sx q[19];
rz(4.784468846870542) q[19];
sx q[19];
rz(11.239114413014018) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.944714902720634) q[19];
rz(0.8520445616456034) q[5];
cx q[10],q[5];
rz(-0.8520445616456034) q[5];
cx q[10],q[5];
rz(0) q[10];
sx q[10];
rz(6.990249923022441) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[6],q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[1];
rz(0) q[1];
sx q[1];
rz(1.8303623570871945) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[21];
sx q[21];
rz(4.452822950092392) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[21],q[1];
rz(-pi/2) q[1];
rz(5.3982644773558786) q[1];
rz(-1.2936357929188602) q[1];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(4.261165153317761) q[21];
cx q[21],q[1];
rz(-4.261165153317761) q[1];
sx q[1];
rz(2.874569055098912) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[21],q[1];
rz(0) q[1];
sx q[1];
rz(3.4086162520806744) q[1];
sx q[1];
rz(14.979578907006001) q[1];
rz(0.5550352997395751) q[1];
sx q[1];
rz(5.575402238642799) q[1];
sx q[1];
rz(8.869742661029804) q[1];
rz(1.8483632363594735) q[1];
rz(pi/2) q[1];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[3],q[6];
rz(-pi/4) q[6];
cx q[3],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[6];
cx q[14],q[6];
x q[14];
rz(0) q[14];
sx q[14];
rz(6.77861087757382) q[14];
sx q[14];
rz(3*pi) q[14];
rz(-pi/2) q[14];
cx q[6],q[8];
id q[6];
id q[6];
rz(5.320230686561337) q[6];
rz(4.754036374721497) q[6];
rz(pi/2) q[8];
rz(6.09407947587304) q[8];
rz(2.3360708099809626) q[8];
rz(pi/2) q[22];
sx q[22];
rz(7.897456096015885) q[22];
sx q[22];
rz(5*pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(3.4836649702660294) q[23];
rz(pi/2) q[23];
cx q[18],q[23];
rz(0) q[18];
sx q[18];
rz(0.06777416343957876) q[18];
sx q[18];
rz(3*pi) q[18];
rz(0) q[23];
sx q[23];
rz(0.06777416343957876) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[18],q[23];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[18],q[22];
rz(pi/4) q[18];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[18],q[22];
rz(-pi/4) q[22];
cx q[18],q[22];
rz(4.320022091631451) q[18];
sx q[18];
rz(2.726283397259845) q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[11];
rz(0) q[11];
sx q[11];
rz(2.3373309468509538) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[18];
sx q[18];
rz(3.9458543603286325) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[11];
rz(-pi/2) q[11];
rz(3.6665857067100913) q[11];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(-1.5058779469423318) q[18];
sx q[18];
rz(2.253277832893691) q[18];
rz(1.4978684719747657) q[18];
cx q[18],q[10];
rz(-1.4978684719747657) q[10];
cx q[18],q[10];
rz(1.4978684719747657) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(1.448390862387758) q[18];
rz(0.5384938399772801) q[18];
sx q[18];
rz(8.373930887791898) q[18];
sx q[18];
rz(8.8862841207921) q[18];
rz(-0.24390699466864274) q[18];
sx q[18];
rz(6.982363252980109) q[18];
sx q[18];
rz(9.668684955438023) q[18];
rz(pi/2) q[18];
sx q[18];
rz(7.638069502039367) q[18];
sx q[18];
rz(5*pi/2) q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.5778654327523451) q[18];
rz(pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[2];
rz(pi/2) q[2];
cx q[2],q[20];
rz(4.947439813697122) q[20];
cx q[2],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0.9914430684538387) q[20];
cx q[22],q[16];
cx q[11],q[22];
rz(pi/2) q[16];
cx q[16],q[21];
cx q[16],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(5.786354755152827) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[22],q[11];
rz(-0.43796416148370876) q[11];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[21],q[22];
rz(2.1004308447680824) q[22];
cx q[21],q[22];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[19];
rz(-2.944714902720634) q[19];
cx q[21],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(0) q[22];
sx q[22];
rz(5.553307966456865) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[14],q[22];
rz(0) q[22];
sx q[22];
rz(0.7298773407227213) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[14],q[22];
rz(-pi/2) q[23];
rz(-3.4836649702660294) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(1.8206386149530984) q[23];
cx q[0],q[23];
rz(-1.8206386149530984) q[23];
cx q[0],q[23];
cx q[0],q[9];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-4.193228102290485) q[23];
rz(pi/2) q[23];
cx q[7],q[23];
rz(0) q[23];
sx q[23];
rz(2.407665910875276) q[23];
sx q[23];
rz(3*pi) q[23];
rz(0) q[7];
sx q[7];
rz(3.8755193963043104) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[23];
rz(-pi/2) q[23];
rz(4.193228102290485) q[23];
cx q[3],q[23];
rz(0.11413995827231979) q[23];
cx q[3],q[23];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[5],q[23];
cx q[23],q[5];
cx q[5],q[23];
rz(5.512515406811957) q[23];
cx q[5],q[23];
rz(0) q[23];
sx q[23];
rz(6.290235909579654) q[23];
sx q[23];
rz(3*pi) q[23];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
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
cx q[17],q[7];
rz(3.2790078548049406) q[7];
cx q[17],q[7];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[12],q[17];
rz(-pi/4) q[17];
cx q[12],q[17];
rz(1.7847412722815628) q[12];
cx q[12],q[4];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0.67435518133886) q[17];
rz(-1.7847412722815628) q[4];
cx q[12],q[4];
cx q[12],q[15];
rz(2.818477177906869) q[12];
sx q[12];
rz(3.772003467209803) q[12];
sx q[12];
rz(13.128709451247538) q[12];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[19],q[15];
rz(2.213295915091543) q[15];
cx q[19],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(1.7847412722815628) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0) q[7];
sx q[7];
rz(4.381938189283242) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(0) q[7];
sx q[7];
rz(1.901247117896344) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
cx q[7],q[17];
rz(-0.67435518133886) q[17];
cx q[7],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
sx q[17];
cx q[1],q[17];
x q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[16],q[1];
rz(5.556157375487726) q[1];
cx q[16],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0.5069487137781946) q[1];
cx q[1],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0) q[17];
sx q[17];
rz(7.660387750300771) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0.5433554642532544) q[17];
rz(-0.5069487137781946) q[19];
cx q[1],q[19];
rz(0.5069487137781946) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[8],q[11];
rz(-2.3360708099809626) q[11];
sx q[11];
rz(0.8215257723472238) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[8],q[11];
rz(0) q[11];
sx q[11];
rz(5.461659534832362) q[11];
sx q[11];
rz(12.19881293223405) q[11];
sx q[11];
rz(5.02073979614295) q[8];
rz(1.6878648691844762) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.9178555727331393) q[8];
cx q[10],q[8];
rz(-0.9178555727331393) q[8];
cx q[10],q[8];
id q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.235878463320864) q[8];
cx q[1],q[8];
rz(-2.235878463320864) q[8];
cx q[1],q[8];
rz(5.09988675305098) q[1];
sx q[1];
rz(3.8403419033543993) q[1];
sx q[1];
rz(15.571315256354362) q[1];
rz(pi/2) q[1];
sx q[1];
rz(1.7569002000934013) q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(1.517299002456197) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[0],q[9];
rz(1.9831515160635755) q[0];
sx q[0];
rz(7.796720852119701) q[0];
sx q[0];
rz(10.558519637551168) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[3];
rz(3.6762281419191014) q[3];
cx q[0],q[3];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[20],q[0];
rz(4.212404428776965) q[0];
cx q[20],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[5];
rz(-pi/2) q[0];
rz(2.0229973417135927) q[0];
cx q[0],q[12];
rz(-2.0229973417135927) q[12];
cx q[0],q[12];
rz(2.0229973417135927) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(2.902424526026688) q[20];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
rz(5.052303902763167) q[2];
sx q[2];
rz(7.525248898921894) q[2];
sx q[2];
rz(10.442586085910145) q[2];
cx q[22],q[2];
cx q[2],q[22];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.14930117527034198) q[3];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(2.066286561509222) q[5];
rz(pi/2) q[5];
cx q[7],q[20];
rz(-2.902424526026688) q[20];
cx q[7],q[20];
cx q[14],q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0.49629632500301124) q[20];
cx q[6],q[20];
rz(-4.754036374721497) q[20];
sx q[20];
rz(2.7890055619153586) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[6],q[20];
rz(0) q[20];
sx q[20];
rz(3.4941797452642276) q[20];
sx q[20];
rz(13.682518010487865) q[20];
cx q[17],q[20];
rz(-0.5433554642532544) q[20];
cx q[17],q[20];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[19],q[17];
rz(0.16282865948467318) q[17];
cx q[19],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
rz(0.4609697796590353) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(0.5433554642532544) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[20];
cx q[0],q[20];
rz(2.1460653042214433) q[0];
rz(-pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(0) q[20];
sx q[20];
rz(3.812245992397024) q[20];
sx q[20];
rz(3*pi) q[20];
rz(1.4107743341507535) q[6];
sx q[6];
rz(9.32640062153115) q[6];
sx q[6];
rz(10.506574575434467) q[6];
rz(-1.7887259827865758) q[6];
rz(5.626075682303134) q[7];
cx q[14],q[7];
rz(-2.2462261514814004) q[14];
rz(pi/2) q[14];
cx q[16],q[14];
rz(0) q[14];
sx q[14];
rz(3.1068791148035357) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[16];
sx q[16];
rz(3.1763061923760505) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[16],q[14];
rz(-pi/2) q[14];
rz(2.2462261514814004) q[14];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[16],q[14];
cx q[0],q[16];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-2.1460653042214433) q[16];
cx q[0],q[16];
x q[0];
rz(-0.3851561178877686) q[0];
rz(pi/2) q[0];
rz(2.1460653042214433) q[16];
rz(-pi/2) q[16];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[2];
rz(3.7570718355270643) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(5.321384330101788) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/4) q[9];
rz(0.6643303836017408) q[9];
rz(-pi/2) q[9];
cx q[13],q[9];
cx q[3],q[13];
rz(-0.14930117527034198) q[13];
cx q[3],q[13];
rz(0.14930117527034198) q[13];
rz(pi/4) q[13];
rz(pi/4) q[13];
cx q[13],q[15];
rz(-pi/4) q[15];
cx q[13],q[15];
sx q[13];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(3.1878496991918985) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[14],q[15];
rz(0.0228612628310496) q[15];
cx q[14],q[15];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.9195139018976596) q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.9088039048071481) q[15];
sx q[15];
rz(3.674181957197437) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[11],q[3];
rz(4.496272054012984) q[3];
cx q[11],q[3];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[5];
rz(0) q[11];
sx q[11];
rz(0.3608857993122183) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi) q[3];
rz(2.157976625533934) q[3];
cx q[3],q[6];
rz(0) q[5];
sx q[5];
rz(0.3608857993122183) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[11],q[5];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(6.193410927457076) q[11];
rz(-4.246557748115284) q[11];
sx q[11];
rz(7.665544710193538) q[11];
sx q[11];
rz(13.671335708884662) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[5];
rz(-2.066286561509222) q[5];
cx q[5],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[5];
cx q[5],q[12];
rz(-pi/4) q[12];
cx q[5],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.771791832548689) q[12];
sx q[12];
rz(3.5140363748251344) q[12];
sx q[12];
rz(12.843841586307205) q[12];
rz(0) q[12];
sx q[12];
rz(3.617311424473455) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(-2.157976625533934) q[6];
sx q[6];
rz(1.2708119304917955) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[3],q[6];
rz(0) q[6];
sx q[6];
rz(5.0123733766877905) q[6];
sx q[6];
rz(13.37148056908989) q[6];
cx q[6],q[10];
cx q[10],q[6];
cx q[10],q[16];
rz(1.2459161073691853) q[10];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[19];
rz(5.3925692938172265) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
x q[6];
cx q[8],q[5];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi) q[5];
rz(pi) q[5];
rz(4.222384551640086) q[5];
rz(3.9871164355669952) q[5];
rz(1.2791825828735126) q[8];
rz(0.4005535770271937) q[8];
rz(pi) q[8];
cx q[8],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[8];
cx q[8],q[16];
rz(-pi/4) q[16];
cx q[8],q[16];
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
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0) q[8];
sx q[8];
rz(4.766305852324688) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(0.9103904330280157) q[9];
cx q[4],q[9];
cx q[4],q[23];
rz(3.19546679937877) q[23];
cx q[4],q[23];
rz(3.035814462605136) q[23];
cx q[22],q[23];
rz(-3.035814462605136) q[23];
cx q[22],q[23];
rz(-pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(3.5867920876968227) q[22];
sx q[22];
rz(5*pi/2) q[22];
rz(-pi/2) q[22];
rz(5.776479010219333) q[23];
sx q[23];
rz(7.580989391306633) q[23];
sx q[23];
rz(12.755680028746843) q[23];
rz(pi/4) q[23];
cx q[23],q[17];
rz(-pi/4) q[17];
cx q[23],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[12];
rz(0) q[12];
sx q[12];
rz(2.6658738827061312) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[17],q[12];
cx q[12],q[6];
rz(2.5957873767090875) q[17];
rz(-1.1694954419879398) q[23];
sx q[23];
rz(2.9710593157373366) q[23];
cx q[15],q[23];
rz(4.019189274348492) q[23];
cx q[15],q[23];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(5.809669753916211) q[23];
sx q[23];
rz(4.657845252131648) q[23];
sx q[23];
rz(10.994783359465886) q[23];
rz(pi/4) q[23];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[18];
rz(-0.5778654327523451) q[18];
cx q[2],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.08972770652524373) q[18];
cx q[18],q[22];
cx q[2],q[10];
rz(-1.2459161073691853) q[10];
cx q[2],q[10];
rz(2.0091046338415968) q[10];
rz(-2.440437078843022) q[2];
rz(pi/2) q[2];
rz(-0.08972770652524373) q[22];
cx q[18],q[22];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.08972770652524373) q[22];
cx q[22],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
sx q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(3.349061405208355) q[15];
sx q[15];
rz(5*pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[1];
cx q[1],q[22];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[22];
rz(pi) q[3];
x q[3];
rz(3.532785377731811) q[3];
cx q[3],q[17];
rz(-3.532785377731811) q[17];
sx q[17];
rz(2.1718479152689527) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[3],q[17];
rz(0) q[17];
sx q[17];
rz(4.1113373919106335) q[17];
sx q[17];
rz(10.361775961792102) q[17];
rz(3.7431382913309705) q[17];
rz(4.684802924657664) q[17];
sx q[17];
rz(5.504390311383425) q[17];
rz(pi/2) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[14];
rz(0) q[14];
sx q[14];
rz(1.8151887454783995) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[4];
sx q[4];
rz(1.8151887454783995) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[14];
rz(-pi/2) q[14];
rz(-0.9195139018976596) q[14];
rz(4.557658944762456) q[14];
sx q[14];
rz(4.933813801980886) q[14];
sx q[14];
rz(13.460371607326278) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-4.575302326425559) q[14];
sx q[14];
rz(8.641658960242825) q[14];
sx q[14];
rz(14.000080287194939) q[14];
rz(5.89597495721542) q[14];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(0.48617261885136803) q[4];
rz(pi/2) q[4];
sx q[4];
rz(4.887103413685205) q[4];
sx q[4];
rz(5*pi/2) q[4];
x q[4];
cx q[5],q[10];
rz(-3.9871164355669952) q[10];
sx q[10];
rz(0.5770934845661908) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[5],q[10];
rz(0) q[10];
sx q[10];
rz(5.706091822613395) q[10];
sx q[10];
rz(11.402789762494779) q[10];
sx q[10];
rz(pi/2) q[5];
sx q[5];
rz(9.11209288762328) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(-1.74501482880306) q[5];
rz(pi/2) q[5];
cx q[6],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.1320900829838307) q[12];
cx q[3],q[12];
rz(-2.1320900829838307) q[12];
cx q[3],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[23],q[3];
rz(-pi/4) q[3];
cx q[23],q[3];
rz(pi/2) q[23];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[22];
rz(pi/2) q[22];
cx q[3],q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(-1.1413279760656194) q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[21],q[9];
rz(pi/4) q[21];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[21],q[9];
rz(-pi/4) q[9];
cx q[21],q[9];
rz(-3.1196907429224208) q[21];
sx q[21];
rz(7.977168342248546) q[21];
sx q[21];
rz(12.5444687036918) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
cx q[21],q[20];
rz(0) q[20];
sx q[20];
rz(2.4709393147825622) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[21],q[20];
sx q[20];
cx q[11],q[20];
x q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[0];
rz(0) q[0];
sx q[0];
rz(1.0031261637108106) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[11];
sx q[11];
rz(5.280059143468776) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[11],q[0];
rz(-pi/2) q[0];
rz(0.3851561178877686) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[2];
rz(0) q[0];
sx q[0];
rz(5.88732120480325) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(0) q[2];
sx q[2];
rz(0.39586410237633674) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[0],q[2];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(5.20223706505808) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/2) q[2];
rz(2.440437078843022) q[2];
cx q[2],q[0];
rz(0) q[0];
sx q[0];
rz(1.0809482421215058) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.1765539327177597) q[0];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[23];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi) q[20];
x q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(3.9717530617828145) q[21];
cx q[23],q[2];
rz(5.228844699633137) q[2];
rz(3.031652330055929) q[2];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-1.8595046577990093) q[9];
sx q[9];
rz(4.751691023022877) q[9];
sx q[9];
rz(11.284282618568389) q[9];
rz(pi/2) q[9];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
id q[13];
rz(pi) q[13];
x q[13];
cx q[13],q[19];
rz(pi/4) q[13];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[13],q[19];
rz(-pi/4) q[19];
cx q[13],q[19];
rz(4.981500047849759) q[13];
cx q[13],q[8];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.8967989299495165) q[19];
cx q[19],q[6];
rz(-2.8967989299495165) q[6];
sx q[6];
rz(2.391163149942416) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[19],q[6];
cx q[19],q[0];
rz(-1.1765539327177597) q[0];
cx q[19],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.049987653203911436) q[0];
id q[19];
rz(2.1275561434029244) q[19];
rz(pi/2) q[19];
rz(0) q[6];
sx q[6];
rz(3.8920221572371703) q[6];
sx q[6];
rz(13.462904866784516) q[6];
rz(-pi/2) q[6];
cx q[17],q[6];
rz(0.9951965484956617) q[17];
cx q[2],q[17];
rz(-3.031652330055929) q[17];
sx q[17];
rz(2.101823152074398) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[2],q[17];
rz(0) q[17];
sx q[17];
rz(4.181362155105188) q[17];
sx q[17];
rz(11.461233742329647) q[17];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(0) q[8];
sx q[8];
rz(1.516879454854898) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[13],q[8];
cx q[0],q[8];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[6],q[13];
rz(-pi/4) q[13];
cx q[6],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.4938879166712644) q[13];
rz(0.19044409241969817) q[6];
rz(-0.049987653203911436) q[8];
cx q[0],q[8];
rz(1.5487734239879078) q[0];
cx q[0],q[1];
rz(-1.5487734239879078) q[1];
cx q[0],q[1];
rz(1.5487734239879078) q[1];
rz(0.049987653203911436) q[8];
cx q[8],q[13];
rz(-1.4938879166712644) q[13];
cx q[8],q[13];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
x q[9];
rz(pi/4) q[9];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
rz(1.7391910198128469) q[7];
cx q[21],q[7];
rz(-3.9717530617828145) q[7];
sx q[7];
rz(0.621803164664068) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[21],q[7];
rz(0.15936948748168742) q[21];
sx q[21];
rz(7.505015176215231) q[21];
sx q[21];
rz(13.31585771120429) q[21];
rz(pi/2) q[21];
cx q[21],q[10];
x q[10];
rz(2.9482351328305896) q[10];
sx q[10];
rz(5.1771881696068665) q[10];
sx q[10];
rz(12.435417127505689) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[2],q[10];
rz(4.39001837887346) q[10];
cx q[2],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[2],q[0];
cx q[0],q[2];
rz(5.2918985969020635) q[2];
sx q[2];
rz(7.115831778824848) q[2];
sx q[2];
rz(13.403750743917504) q[2];
x q[21];
rz(0.6405891553143371) q[21];
sx q[21];
rz(3.9661507369387605) q[21];
sx q[21];
rz(8.784188805455042) q[21];
rz(0) q[7];
sx q[7];
rz(5.661382142515518) q[7];
sx q[7];
rz(11.657340002739346) q[7];
rz(3.8417823577316996) q[7];
sx q[7];
rz(4.152746070211876) q[7];
sx q[7];
rz(13.51364271383619) q[7];
rz(4.54715427279826) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
cx q[9],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[9];
cx q[9],q[18];
rz(-pi/4) q[18];
cx q[9],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[20],q[18];
cx q[18],q[20];
cx q[20],q[18];
rz(pi/4) q[18];
cx q[18],q[12];
rz(-pi/4) q[12];
cx q[18],q[12];
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
x q[18];
rz(pi/4) q[20];
cx q[20],q[16];
rz(-pi/4) q[16];
cx q[20],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[21];
cx q[21],q[16];
rz(4.037396551354986) q[16];
rz(2.4520399613853057) q[16];
cx q[16],q[6];
cx q[23],q[12];
rz(4.436254650321649) q[12];
cx q[23],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/4) q[23];
rz(-2.4520399613853057) q[6];
sx q[6];
rz(1.5013722625796921) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[16],q[6];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
cx q[13],q[16];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(0) q[6];
sx q[6];
rz(4.781813044599894) q[6];
sx q[6];
rz(11.686373829734986) q[6];
rz(3.1089942664276142) q[6];
cx q[7],q[20];
rz(4.3135180496679695) q[20];
cx q[7],q[20];
cx q[20],q[15];
rz(1.992218793029325) q[15];
cx q[20],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[19];
cx q[19],q[15];
rz(0.13570863088702803) q[15];
sx q[15];
rz(9.294130365119086) q[15];
sx q[15];
rz(14.330664436459244) q[15];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.5899026976389081) q[19];
rz(pi/2) q[20];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[21],q[7];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[20];
cx q[20],q[8];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[0];
cx q[0],q[20];
cx q[20],q[0];
rz(1.8908478735856937) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(pi/4) q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(-pi/4) q[9];
cx q[11],q[9];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[5];
rz(0) q[11];
sx q[11];
rz(3.199784164091512) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[5];
sx q[5];
rz(3.083401143088074) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[11],q[5];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi) q[11];
x q[11];
cx q[11],q[12];
rz(pi) q[11];
x q[11];
rz(pi/2) q[12];
rz(0) q[12];
sx q[12];
rz(4.172493658617723) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[12],q[13];
rz(0.4061082040215627) q[13];
cx q[12],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[5];
rz(1.74501482880306) q[5];
rz(1.8057480014092928) q[5];
cx q[5],q[22];
rz(-1.8057480014092928) q[22];
cx q[5],q[22];
cx q[17],q[5];
rz(1.8057480014092928) q[22];
cx q[22],q[23];
rz(-pi/4) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(-pi/4) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi/2) q[23];
cx q[23],q[6];
rz(5.130743579083455) q[5];
cx q[17],q[5];
rz(1.3852865456329537) q[17];
cx q[1],q[17];
rz(-1.3852865456329537) q[17];
cx q[1],q[17];
rz(-pi/2) q[1];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[10],q[5];
rz(4.653966356650031) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[16],q[10];
rz(3.7169817573874115) q[10];
cx q[16],q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-2.9915781418658383) q[5];
rz(pi/2) q[5];
rz(-3.1089942664276142) q[6];
cx q[23],q[6];
cx q[23],q[17];
cx q[17],q[23];
cx q[8],q[11];
rz(-1.8908478735856937) q[11];
cx q[8],q[11];
rz(1.8908478735856937) q[11];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(2.8452357946086195) q[9];
cx q[4],q[9];
rz(-2.8452357946086195) q[9];
cx q[4],q[9];
rz(1.8635445178430237) q[4];
sx q[4];
rz(1.6521697370334338) q[4];
cx q[3],q[4];
cx q[3],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(2.531068534039471) q[21];
cx q[3],q[21];
rz(-2.531068534039471) q[21];
cx q[3],q[21];
rz(1.4641857682788042) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[14],q[9];
rz(6.168939109506517) q[9];
cx q[14],q[9];
rz(pi) q[14];
x q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[7],q[14];
rz(2.1804962097460234) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[5];
rz(0) q[14];
sx q[14];
rz(3.2664393877663405) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[5];
sx q[5];
rz(3.0167459194132458) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[14],q[5];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[5];
rz(2.9915781418658383) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(pi/4) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[6],q[7];
rz(-pi/4) q[7];
cx q[6],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[18];
rz(3.8218272953359618) q[18];
cx q[9],q[18];
rz(2.5710155139151767) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(2.757918064180936) q[9];
cx q[4],q[9];
rz(-2.757918064180936) q[9];
cx q[4],q[9];
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
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];