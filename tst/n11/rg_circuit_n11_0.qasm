OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.6324757505804937) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[5];
cx q[2],q[6];
rz(-2.6324757505804937) q[6];
cx q[2],q[6];
rz(1.8183958909505382) q[2];
sx q[2];
rz(3.9545840542387087) q[2];
sx q[2];
rz(14.844657085355951) q[2];
rz(-pi/2) q[2];
rz(2.6324757505804937) q[6];
rz(1.971614015569729) q[6];
cx q[5],q[6];
rz(-1.971614015569729) q[6];
cx q[5],q[6];
rz(-pi/2) q[6];
cx q[2],q[6];
cx q[6],q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
rz(pi/2) q[6];
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
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[0],q[7];
rz(2.3336716959189623) q[7];
cx q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-0.6220558251942233) q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[3];
cx q[3],q[8];
rz(pi/4) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(-pi/4) q[8];
cx q[3],q[8];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[1];
rz(3.533131699587626) q[1];
cx q[9],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(6.8294611127107405) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(-pi/2) q[1];
cx q[5],q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/4) q[5];
rz(3.2874610722604785) q[5];
rz(pi/2) q[5];
sx q[5];
rz(6.017946191914783) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(-4.231848811393963) q[5];
sx q[5];
rz(8.912038386142607) q[5];
sx q[5];
rz(13.656626772163342) q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.3470359071437984) q[9];
cx q[9],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[9],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
id q[8];
cx q[8],q[2];
rz(-pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(0.3238600659023607) q[8];
cx q[9],q[1];
rz(3.0316649778344886) q[1];
cx q[9],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.8814387930334786) q[9];
sx q[9];
rz(2.2728376142450593) q[9];
x q[9];
rz(pi/4) q[9];
cx q[10],q[4];
rz(pi/4) q[10];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[10],q[4];
rz(-pi/4) q[4];
cx q[10],q[4];
rz(5.207913732071936) q[10];
rz(1.8376527797066062) q[10];
cx q[10],q[0];
rz(-1.8376527797066062) q[0];
sx q[0];
rz(3.0672561617673426) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[10],q[0];
rz(0) q[0];
sx q[0];
rz(3.2159291454122436) q[0];
sx q[0];
rz(11.884486565670208) q[0];
x q[0];
x q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
rz(pi/4) q[0];
rz(5.214831421944569) q[0];
sx q[0];
rz(4.346169970605912) q[0];
sx q[0];
rz(12.106439183416649) q[0];
rz(4.4314724216521935) q[1];
sx q[1];
rz(6.082567972289377) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[10];
rz(0.21826768465509228) q[10];
rz(2.666771957955384) q[10];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.5451012329376574) q[4];
cx q[7],q[4];
rz(-1.5451012329376574) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(6.026724457355674) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-0.380129701848436) q[4];
cx q[10],q[4];
rz(-2.666771957955384) q[4];
sx q[4];
rz(0.42332119489438425) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[10],q[4];
rz(0) q[10];
sx q[10];
rz(4.054260054883727) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[4];
sx q[4];
rz(5.8598641122852015) q[4];
sx q[4];
rz(12.4716796205732) q[4];
rz(2.0949803747718745) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[3],q[7];
rz(3.119727393307877) q[7];
cx q[3],q[7];
cx q[3],q[10];
rz(0) q[10];
sx q[10];
rz(2.2289252522958587) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[3],q[10];
rz(1.429838300742481) q[10];
sx q[10];
rz(5.7762825106263955) q[10];
sx q[10];
rz(14.688966005309855) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi) q[10];
x q[10];
rz(pi/2) q[10];
sx q[10];
rz(4.295491849456812) q[10];
sx q[10];
rz(5*pi/2) q[10];
x q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[7];
rz(-2.0949803747718745) q[7];
cx q[4],q[7];
rz(-pi/2) q[4];
cx q[2],q[4];
rz(4.38433215525242) q[2];
sx q[2];
rz(3.972222180562583) q[2];
sx q[2];
rz(10.986708491587127) q[2];
rz(3.7203262650069355) q[2];
sx q[2];
rz(7.17796175676659) q[2];
rz(pi/2) q[2];
sx q[2];
rz(4.153735423237832) q[2];
sx q[2];
rz(5*pi/2) q[2];
rz(0) q[2];
sx q[2];
rz(5.02365460136328) q[2];
sx q[2];
rz(3*pi) q[2];
rz(1.0174862118884942) q[2];
rz(pi/2) q[4];
rz(-1.2250921816342508) q[4];
rz(pi/2) q[4];
cx q[1],q[4];
rz(0) q[1];
sx q[1];
rz(3.5690502125068417) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[4];
sx q[4];
rz(2.7141350946727445) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[1],q[4];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(1.336943388276346) q[1];
cx q[0],q[1];
rz(-1.336943388276346) q[1];
cx q[0],q[1];
id q[1];
rz(2.4238691555438807) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[4];
rz(1.2250921816342508) q[4];
cx q[6],q[4];
rz(-pi/2) q[4];
cx q[0],q[4];
rz(0.9012905558418776) q[0];
cx q[0],q[2];
rz(-1.0174862118884942) q[2];
cx q[0],q[2];
rz(-pi/2) q[0];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(-3.333305678166756) q[6];
sx q[6];
rz(3.207932769687628) q[6];
sx q[6];
rz(12.758083638936135) q[6];
cx q[6],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[6];
cx q[6],q[5];
rz(-pi/4) q[5];
cx q[6],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-0.7028073842852538) q[6];
rz(pi/2) q[6];
cx q[1],q[6];
rz(0) q[1];
sx q[1];
rz(4.97636498775338) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[6];
sx q[6];
rz(1.306820319426207) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[1],q[6];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
id q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[6];
rz(0.7028073842852538) q[6];
cx q[6],q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.6732587010327773) q[6];
rz(2.0949803747718745) q[7];
cx q[7],q[8];
rz(-0.3238600659023607) q[8];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[3];
rz(0.41130403329274273) q[3];
cx q[7],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[10],q[7];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[2],q[10];
rz(0.6461830501954433) q[10];
cx q[2],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[10];
sx q[10];
rz(4.514655548636845) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
x q[2];
x q[2];
rz(pi) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[9],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.5415312900282212) q[8];
sx q[8];
rz(4.858104383614308) q[8];
sx q[8];
rz(15.057199678165318) q[8];
x q[8];
rz(0.17580293504669925) q[8];
cx q[4],q[8];
rz(-0.17580293504669925) q[8];
cx q[4],q[8];
rz(1.286343044819198) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0) q[8];
sx q[8];
rz(7.2924138707208375) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[3];
rz(2.332716782427179) q[3];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-1.3639837818049656) q[3];
sx q[3];
rz(2.920339393368727) q[3];
cx q[7],q[3];
x q[3];
rz(1.2211483260889002) q[3];
sx q[3];
rz(7.74568996650792) q[3];
x q[3];
rz(-pi/2) q[3];
rz(pi/2) q[7];
cx q[5],q[7];
cx q[7],q[5];
cx q[5],q[0];
rz(1.135748900711204) q[0];
cx q[5],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[10];
rz(1.2883142021194005) q[10];
cx q[0],q[10];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[3];
rz(-1.7432781792484062) q[10];
rz(pi/2) q[10];
rz(pi/2) q[3];
rz(pi) q[3];
x q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.500185477123566) q[7];
sx q[7];
rz(6.263130363889051) q[7];
sx q[7];
rz(14.58541337384299) q[7];
x q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(0) q[9];
sx q[9];
rz(8.601611548025492) q[9];
sx q[9];
rz(3*pi) q[9];
rz(pi) q[9];
x q[9];
rz(pi) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(3.134289440094848) q[9];
cx q[4],q[9];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[8];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[5],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.1869028718440529) q[4];
cx q[4],q[0];
rz(-1.1869028718440529) q[0];
cx q[4],q[0];
rz(1.1869028718440529) q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[8];
rz(-1.6732587010327773) q[8];
cx q[6],q[8];
rz(5.009351225136061) q[6];
sx q[6];
rz(4.620353900620016) q[6];
sx q[6];
rz(10.997389643381753) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[10];
rz(0) q[10];
sx q[10];
rz(1.6394118506970519) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[6];
sx q[6];
rz(4.643773456482535) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[10];
rz(-pi/2) q[10];
rz(1.7432781792484062) q[10];
rz(pi) q[10];
x q[10];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(1.6732587010327773) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[7];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.226938011347519) q[7];
sx q[7];
rz(7.4061434103711) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[10],q[7];
rz(0.6964578200359289) q[7];
cx q[10],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(5.097337736040623) q[7];
sx q[7];
rz(8.511557259845564) q[7];
sx q[7];
rz(13.442056819947982) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
rz(1.9001782763931134) q[4];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.6197128986782505) q[4];
sx q[4];
rz(5.708789253687904) q[4];
sx q[4];
rz(12.954762873836955) q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-6.2364735060214045) q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(0) q[1];
sx q[1];
rz(5.3218526582325545) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[9];
sx q[9];
rz(0.9613326489470317) q[9];
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
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[5];
cx q[5],q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[2],q[1];
rz(1.4620240594829281) q[1];
cx q[2],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
x q[1];
cx q[1],q[4];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
cx q[2],q[0];
rz(3.673305299903034) q[0];
cx q[2],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(0.06566477181111376) q[0];
rz(pi) q[0];
x q[0];
cx q[2],q[6];
cx q[4],q[1];
cx q[1],q[4];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(1.1488052821830803) q[1];
sx q[1];
rz(8.923064751547962) q[1];
sx q[1];
rz(8.2759726785863) q[1];
rz(2.7041014159417864) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.3743680464466397) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(3.055570549215354) q[6];
cx q[2],q[6];
rz(-pi/4) q[2];
rz(pi/2) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[5];
rz(-pi/4) q[5];
cx q[8],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(-3.6569487533684075) q[8];
sx q[8];
rz(6.973979279975222) q[8];
sx q[8];
rz(13.081726714137787) q[8];
cx q[4],q[8];
rz(-2.7041014159417864) q[8];
cx q[4],q[8];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.7041014159417864) q[8];
sx q[8];
rz(-pi/2) q[9];
rz(6.2364735060214045) q[9];
sx q[9];
rz(6.1815789585939305) q[9];
rz(0) q[9];
sx q[9];
rz(8.004986083565676) q[9];
sx q[9];
rz(3*pi) q[9];
rz(pi/4) q[9];
cx q[9],q[3];
rz(-pi/4) q[3];
cx q[9],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[5];
rz(pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[3];
cx q[3],q[6];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-6.177345588268219) q[3];
rz(pi/2) q[3];
cx q[5],q[3];
rz(0) q[3];
sx q[3];
rz(2.5397871764357305) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[5];
sx q[5];
rz(3.7433981307438557) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[3];
rz(-pi/2) q[3];
rz(6.177345588268219) q[3];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
cx q[4],q[6];
cx q[6],q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
id q[9];
rz(0.8313766083943342) q[9];
cx q[9],q[10];
rz(-0.8313766083943342) q[10];
cx q[9],q[10];
rz(0.8313766083943342) q[10];
rz(pi/2) q[10];
cx q[10],q[8];
x q[10];
sx q[9];
cx q[2],q[9];
x q[2];
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