OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
rz(6.183282743155809) q[0];
id q[1];
rz(5.072874790760689) q[1];
sx q[1];
rz(4.024617650722533) q[1];
sx q[1];
rz(12.78850767512359) q[1];
rz(0) q[3];
sx q[3];
rz(7.771775765001868) q[3];
sx q[3];
rz(3*pi) q[3];
id q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.7696493411243837) q[3];
rz(0.5748493795288714) q[4];
rz(pi/2) q[5];
rz(pi) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.1025674771463656) q[7];
cx q[2],q[7];
rz(-2.1025674771463656) q[7];
cx q[2],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.6568086011474366) q[7];
rz(pi) q[8];
x q[8];
rz(pi) q[8];
x q[8];
rz(3.7588968594067214) q[8];
rz(pi/4) q[9];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[9],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
id q[6];
rz(2.3894343067290884) q[6];
rz(-pi/2) q[9];
rz(0.0883113418003384) q[9];
cx q[5],q[9];
rz(-0.0883113418003384) q[9];
cx q[5],q[9];
cx q[5],q[3];
rz(-2.7696493411243837) q[3];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-3.5770872501724926) q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[10];
cx q[10],q[7];
rz(-2.6568086011474366) q[7];
cx q[10],q[7];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[2];
rz(5.900006176704842) q[2];
cx q[10],q[2];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(7.654287243808133) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[5];
rz(0) q[10];
sx q[10];
rz(6.11462409808243) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[8];
rz(0) q[5];
sx q[5];
rz(0.16856120909715555) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[10],q[5];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi) q[10];
x q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[5];
rz(3.5770872501724926) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[2];
rz(1.2972029281624808) q[2];
rz(0) q[8];
sx q[8];
rz(8.423484030392896) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[8];
sx q[8];
rz(3.187082173665374) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(pi/2) q[8];
rz(2.8994665738674277) q[11];
cx q[11],q[4];
rz(-2.8994665738674277) q[4];
sx q[4];
rz(1.209530451834918) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[11],q[4];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[1];
rz(2.2646282958676553) q[1];
cx q[11],q[1];
cx q[1],q[6];
cx q[11],q[9];
rz(pi/4) q[11];
rz(0) q[4];
sx q[4];
rz(5.073654855344668) q[4];
sx q[4];
rz(11.749395155107935) q[4];
cx q[4],q[0];
rz(2.860722743940744) q[0];
cx q[4],q[0];
rz(1.6559466367154005) q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(-2.3894343067290884) q[6];
cx q[1],q[6];
rz(4.384362568088542) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[7],q[4];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(-pi/2) q[4];
rz(-6.028925562807516) q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(0) q[3];
sx q[3];
rz(5.513271012173115) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[4];
sx q[4];
rz(0.7699142950064708) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(4.4713665710361035) q[3];
sx q[3];
rz(3*pi) q[3];
rz(-pi/2) q[4];
rz(6.028925562807516) q[4];
cx q[4],q[3];
rz(0) q[3];
sx q[3];
rz(1.8118187361434823) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[4],q[3];
sx q[3];
rz(pi/2) q[4];
sx q[4];
rz(9.409569924322113) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(0.36081332073735806) q[4];
sx q[4];
rz(5.972615827763453) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[2];
cx q[2],q[0];
rz(-pi/4) q[0];
cx q[2],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.722614173192282) q[0];
rz(pi/2) q[0];
rz(pi) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.052729168496461) q[7];
cx q[8],q[3];
x q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[11],q[9];
rz(-pi/4) q[9];
cx q[11],q[9];
rz(pi/2) q[11];
sx q[11];
rz(8.805047132144932) q[11];
sx q[11];
rz(5*pi/2) q[11];
cx q[11],q[6];
rz(1.8207657983849972) q[11];
cx q[11],q[2];
rz(-1.8207657983849972) q[2];
cx q[11],q[2];
rz(4.133785795910081) q[11];
rz(1.8207657983849972) q[2];
rz(4.130168046461811) q[2];
sx q[2];
rz(5.152460466509696) q[2];
rz(2.8893121488763116) q[2];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(2.9841844433284073) q[6];
cx q[10],q[6];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(6.263391627469492) q[6];
sx q[6];
rz(6.419509460058462) q[6];
sx q[6];
rz(11.964001608200643) q[6];
rz(0) q[6];
sx q[6];
rz(5.509628817053985) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[4],q[6];
rz(0) q[6];
sx q[6];
rz(0.7735564901256011) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[4],q[6];
rz(2.739807999899114) q[4];
rz(-pi/2) q[6];
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
cx q[1],q[9];
rz(5.097577221080657) q[9];
cx q[1],q[9];
sx q[1];
cx q[1],q[7];
rz(-1.052729168496461) q[7];
cx q[1],q[7];
rz(1.972512029923571) q[1];
cx q[11],q[1];
rz(-4.133785795910081) q[1];
sx q[1];
rz(2.1960869699859495) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[11],q[1];
rz(0) q[1];
sx q[1];
rz(4.087098337193637) q[1];
sx q[1];
rz(11.58605172675589) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[4],q[1];
rz(-2.739807999899114) q[1];
cx q[4],q[1];
rz(2.739807999899114) q[1];
rz(0.24631515837259976) q[1];
sx q[1];
rz(8.744907065696376) q[1];
sx q[1];
rz(9.17846280239678) q[1];
rz(pi) q[1];
x q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.131068462519359) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[3],q[7];
rz(3.6992745502759927) q[7];
cx q[3],q[7];
rz(0.7774173805246126) q[3];
sx q[3];
rz(5.7565684740250544) q[3];
sx q[3];
rz(8.647360580244767) q[3];
rz(pi/2) q[3];
sx q[3];
rz(7.47508103310953) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(5.897515982354756) q[3];
rz(2.390141496061122) q[3];
cx q[3],q[4];
rz(-2.390141496061122) q[4];
sx q[4];
rz(0.13795171720520827) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[4];
sx q[4];
rz(6.145233589974378) q[4];
sx q[4];
rz(11.683850994311143) q[4];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(4.757777639437909) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(3.336132024736801) q[7];
sx q[7];
rz(3.690804577671578) q[7];
sx q[7];
rz(15.352998358723125) q[7];
rz(pi/2) q[7];
rz(1.3829990269956645) q[7];
sx q[7];
rz(6.40589722404106) q[7];
sx q[7];
rz(10.561692911698463) q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(0.6333761609447857) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[5],q[9];
rz(2.694570122659997) q[9];
cx q[5],q[9];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[0];
rz(0) q[0];
sx q[0];
rz(0.0175821157771332) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[5];
sx q[5];
rz(0.0175821157771332) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[0];
rz(-pi/2) q[0];
rz(-3.722614173192282) q[0];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(2.2273157794145346) q[5];
rz(pi/2) q[5];
cx q[8],q[5];
rz(0) q[5];
sx q[5];
rz(0.2517924155278486) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[8];
sx q[8];
rz(0.2517924155278486) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[5];
rz(-pi/2) q[5];
rz(-2.2273157794145346) q[5];
rz(0.017340339300207973) q[5];
rz(2.695051832934552) q[5];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(2.054922445975357) q[8];
cx q[8],q[11];
rz(-2.054922445975357) q[11];
cx q[8],q[11];
rz(2.054922445975357) q[11];
cx q[11],q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
cx q[8],q[2];
cx q[2],q[8];
cx q[8],q[2];
sx q[2];
rz(-4.355060500185311) q[2];
sx q[2];
rz(4.474554440067427) q[2];
sx q[2];
rz(13.77983846095469) q[2];
rz(1.0256675577414145) q[2];
rz(-pi/4) q[2];
rz(0.08675202983799579) q[2];
sx q[2];
rz(7.1992041917035) q[2];
rz(pi/4) q[2];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
sx q[9];
rz(-pi/2) q[9];
cx q[0],q[9];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[10],q[0];
rz(-pi/4) q[0];
cx q[10],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(4.749782286372499) q[0];
sx q[0];
rz(4.5819270314196) q[0];
sx q[0];
rz(14.53077552735106) q[0];
rz(-0.6181979420705339) q[0];
sx q[0];
rz(5.3701741550893) q[0];
sx q[0];
rz(10.042975902839913) q[0];
rz(pi/2) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[8],q[10];
rz(2.2321792419771485) q[10];
cx q[8],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
cx q[10],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[0],q[8];
rz(0.5863815590744063) q[0];
rz(pi/2) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[0];
rz(0) q[0];
sx q[0];
rz(2.148036813641636) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[8];
sx q[8];
rz(2.148036813641636) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[0];
rz(-pi/2) q[0];
rz(-0.5863815590744063) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[10],q[0];
rz(2.5087514705068217) q[0];
cx q[10],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.1167920009667838) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.22997593001125283) q[10];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(4.975995915227492) q[8];
sx q[8];
rz(5*pi/2) q[8];
cx q[10],q[8];
rz(-0.22997593001125283) q[8];
cx q[10],q[8];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.22997593001125283) q[8];
cx q[7],q[8];
rz(-0.6333761609447857) q[8];
cx q[7],q[8];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.6333761609447857) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[5],q[9];
rz(-2.695051832934552) q[9];
cx q[5],q[9];
rz(-pi/2) q[5];
cx q[11],q[5];
rz(-2.673808956919538) q[11];
rz(pi/2) q[11];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(0.2560441124906614) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[3];
sx q[3];
rz(6.027141194688925) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[11];
rz(-pi/2) q[11];
rz(2.673808956919538) q[11];
rz(-4.428825166052507) q[11];
sx q[11];
rz(3.9755548947322827) q[11];
sx q[11];
rz(13.853603126821886) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[3],q[4];
rz(-pi/4) q[4];
cx q[3],q[4];
rz(1.145740883965835) q[3];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.837068143066252) q[4];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.695051832934552) q[9];
cx q[9],q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
rz(4.570923724270101) q[5];
cx q[6],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
cx q[5],q[6];
x q[5];
cx q[5],q[4];
rz(-2.837068143066252) q[4];
cx q[5],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(3.3477561410069647) q[5];
rz(pi) q[5];
rz(pi/4) q[5];
cx q[5],q[10];
rz(-pi/4) q[10];
cx q[5],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
id q[5];
rz(0.757935079789666) q[5];
cx q[6],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[3],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[2],q[11];
rz(-pi/4) q[11];
cx q[2],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[11];
cx q[2],q[4];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[11],q[2];
rz(1.798627110665533) q[2];
cx q[11],q[2];
rz(0.6175986506307608) q[11];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(5.2451676002969325) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.6022287872493277) q[3];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
cx q[6],q[0];
rz(-2.1167920009667838) q[0];
cx q[6],q[0];
id q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.6146728647749948) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[0];
rz(-1.6146728647749948) q[0];
cx q[8],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[8],q[3];
rz(-2.6022287872493277) q[3];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(4.134365977174346) q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(1.3128740045199334) q[9];
sx q[9];
rz(7.721368062717045) q[9];
sx q[9];
rz(10.78057088190019) q[9];
rz(pi/2) q[9];
sx q[9];
rz(5.1436163101957675) q[9];
sx q[9];
rz(5*pi/2) q[9];
cx q[1],q[9];
rz(1.966342670221642) q[9];
cx q[1],q[9];
rz(4.8343627795416735) q[1];
rz(pi/4) q[1];
cx q[1],q[6];
rz(-pi/4) q[6];
cx q[1],q[6];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-2.830322242158939) q[6];
rz(pi/2) q[6];
cx q[7],q[6];
rz(0) q[6];
sx q[6];
rz(1.6347597776845295) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[7];
sx q[7];
rz(4.648425529495057) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[6];
rz(-pi/2) q[6];
rz(2.830322242158939) q[6];
rz(-2.263643834607731) q[6];
sx q[6];
rz(5.318926081925866) q[6];
sx q[6];
rz(11.68842179537711) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
cx q[6],q[5];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(2.9723269337265448) q[7];
cx q[0],q[7];
rz(-2.9723269337265448) q[7];
cx q[0],q[7];
cx q[11],q[0];
rz(-0.6175986506307608) q[0];
cx q[11],q[0];
rz(0.6175986506307608) q[0];
rz(3.611306552103474) q[0];
rz(3.110494986488633) q[11];
rz(pi/2) q[11];
rz(pi) q[7];
x q[7];
rz(-pi/2) q[7];
rz(-pi/4) q[9];
rz(pi/4) q[9];
rz(-6.059376397987695) q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(0) q[1];
sx q[1];
rz(4.888913883663322) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[9];
sx q[9];
rz(1.3942714235162643) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[1],q[9];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[10];
rz(0.8199828150643738) q[10];
cx q[1],q[10];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[3];
rz(0) q[1];
sx q[1];
rz(1.364796257496233) q[1];
sx q[1];
rz(3*pi) q[1];
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
rz(pi/4) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0) q[3];
sx q[3];
rz(1.364796257496233) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[3];
rz(-4.134365977174346) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(2.021672911153333) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[3];
sx q[3];
rz(2.021672911153333) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[11];
rz(-pi/2) q[11];
rz(-3.110494986488633) q[11];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/2) q[9];
rz(6.059376397987695) q[9];
cx q[9],q[4];
rz(1.173624420931177) q[4];
cx q[9],q[4];
sx q[4];
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
cx q[9],q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[8],q[1];
rz(pi/2) q[1];
cx q[9],q[7];
rz(pi/2) q[7];
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