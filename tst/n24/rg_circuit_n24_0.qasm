OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
rz(1.1736559147561278) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[5];
cx q[5],q[6];
rz(1.2361983830504135) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.6531310071528242) q[6];
rz(-pi/4) q[7];
rz(pi/4) q[7];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
rz(1.701073910036171) q[9];
sx q[9];
rz(6.971091505152767) q[9];
sx q[9];
rz(12.423641379732238) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[7],q[9];
rz(-pi/4) q[9];
cx q[7],q[9];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-1.120406025989464) q[9];
rz(-pi/2) q[10];
cx q[1],q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[11];
rz(pi/4) q[11];
rz(3.445532709466569) q[12];
rz(2.2471704617140107) q[12];
cx q[13],q[0];
cx q[0],q[13];
rz(2.879849514151662) q[0];
cx q[0],q[5];
cx q[13],q[6];
rz(-2.879849514151662) q[5];
sx q[5];
rz(0.7815986948430074) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[0],q[5];
x q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[5];
sx q[5];
rz(5.501586612336579) q[5];
sx q[5];
rz(11.068429091870627) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-2.6531310071528242) q[6];
cx q[13],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[7];
rz(-0.2329973316553462) q[7];
sx q[7];
rz(3.9591138211773145) q[7];
sx q[7];
rz(9.657775292424725) q[7];
rz(pi/2) q[7];
rz(pi/4) q[16];
cx q[16],q[14];
rz(-pi/4) q[14];
cx q[16],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.2161884210264753) q[14];
rz(-3.9990280029888963) q[16];
rz(pi/2) q[16];
cx q[10],q[16];
rz(0) q[10];
sx q[10];
rz(5.05437087563238) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[16];
sx q[16];
rz(1.2288144315472056) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[10],q[16];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
cx q[10],q[5];
rz(-pi/2) q[16];
rz(3.9990280029888963) q[16];
rz(pi/4) q[16];
rz(3.4374599756482023) q[16];
sx q[16];
rz(4.57200972951952) q[16];
sx q[16];
rz(15.111062566760632) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[7];
rz(-pi/4) q[5];
cx q[10],q[5];
rz(-pi/2) q[10];
cx q[15],q[10];
rz(pi/2) q[10];
rz(1.3371595468678903) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[15];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
rz(pi/4) q[5];
cx q[7],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(0) q[17];
sx q[17];
rz(5.2915372617650185) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[18],q[3];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[3];
rz(-2.8789177086880846) q[3];
sx q[3];
rz(8.019117610238396) q[3];
sx q[3];
rz(12.303695669457465) q[3];
rz(pi) q[3];
x q[3];
rz(0.4586371613956387) q[3];
cx q[2],q[19];
rz(-1.1736559147561278) q[19];
cx q[2],q[19];
rz(1.1736559147561278) q[19];
rz(pi/4) q[19];
cx q[19],q[18];
rz(-pi/4) q[18];
cx q[19],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(6.175099809899768) q[18];
sx q[18];
rz(7.087231206279965) q[18];
sx q[18];
rz(15.530676071629077) q[18];
rz(3.576811576988774) q[18];
rz(3.051375506801725) q[19];
cx q[19],q[9];
rz(pi/4) q[2];
rz(-0.02561608201404919) q[2];
rz(-3.051375506801725) q[9];
sx q[9];
rz(1.350598418463389) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[19],q[9];
rz(0) q[9];
sx q[9];
rz(4.932586888716197) q[9];
sx q[9];
rz(13.59655949356057) q[9];
rz(-pi/4) q[9];
rz(0.930069526709785) q[9];
rz(1.5300094863739437) q[20];
rz(pi/2) q[20];
cx q[4],q[20];
rz(0) q[20];
sx q[20];
rz(2.7975984543238748) q[20];
sx q[20];
rz(3*pi) q[20];
rz(0) q[4];
sx q[4];
rz(2.7975984543238748) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[20];
rz(-pi/2) q[20];
rz(-1.5300094863739437) q[20];
rz(3.3541315619755223) q[20];
cx q[20],q[6];
sx q[20];
cx q[3],q[20];
rz(-0.4586371613956387) q[20];
cx q[3],q[20];
rz(0.4586371613956387) q[20];
rz(pi) q[20];
rz(pi/2) q[20];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-5.250528482403461) q[4];
rz(pi/2) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(3.0846773087614503) q[21];
sx q[21];
rz(1.642316003236615) q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[4];
rz(0) q[21];
sx q[21];
rz(3.2079330962529573) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[4];
sx q[4];
rz(3.075252210926629) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[21],q[4];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(0) q[21];
sx q[21];
rz(5.294442794281152) q[21];
sx q[21];
rz(3*pi) q[21];
rz(-pi/2) q[4];
rz(5.250528482403461) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[6];
cx q[6],q[4];
rz(3.7219454805607612) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[22],q[8];
rz(0) q[22];
sx q[22];
rz(3.249929289057933) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[1],q[22];
rz(0) q[22];
sx q[22];
rz(3.0332560181216532) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[1],q[22];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[14],q[8];
rz(-2.2161884210264753) q[8];
cx q[14],q[8];
rz(2.2161884210264753) q[8];
cx q[1],q[8];
rz(1.3628196050586205) q[1];
cx q[1],q[21];
rz(-1.3628196050586205) q[21];
cx q[1],q[21];
rz(4.714758483816876) q[1];
sx q[1];
rz(6.518768511211231) q[1];
sx q[1];
rz(12.34348197494415) q[1];
rz(pi/2) q[1];
rz(1.3628196050586205) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(-1.6454854693645724) q[8];
cx q[18],q[8];
rz(-3.576811576988774) q[8];
sx q[8];
rz(0.5972268281452191) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[18],q[8];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0) q[8];
sx q[8];
rz(5.685958479034367) q[8];
sx q[8];
rz(14.647075007122726) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[10];
rz(0) q[10];
sx q[10];
rz(0.18180577132202425) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[8];
sx q[8];
rz(0.18180577132202425) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[10];
rz(-pi/2) q[10];
rz(-1.3371595468678903) q[10];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(8.167882436516598) q[8];
sx q[8];
rz(3*pi) q[8];
rz(-1.1964587130073348) q[23];
cx q[12],q[23];
rz(-2.2471704617140107) q[23];
sx q[23];
rz(1.172307866840365) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[12],q[23];
cx q[12],q[17];
rz(0) q[17];
sx q[17];
rz(0.9916480454145682) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[12],q[17];
cx q[14],q[12];
cx q[12],q[14];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.9546017711244363) q[14];
rz(5.950722026684592) q[17];
rz(5.9754081072432825) q[17];
cx q[17],q[2];
cx q[19],q[14];
rz(-0.9546017711244363) q[14];
cx q[19],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[19],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0) q[18];
sx q[18];
rz(4.784314410974534) q[18];
sx q[18];
rz(3*pi) q[18];
rz(pi/2) q[19];
cx q[19],q[21];
x q[19];
rz(2.6389429258236925) q[19];
rz(-5.9754081072432825) q[2];
sx q[2];
rz(1.9323914148737482) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[17],q[2];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0) q[2];
sx q[2];
rz(4.350793892305838) q[2];
sx q[2];
rz(15.425802150026712) q[2];
rz(4.347150401468653) q[2];
sx q[2];
rz(6.350695898943199) q[2];
sx q[2];
rz(10.03361541926549) q[2];
rz(0) q[23];
sx q[23];
rz(5.110877440339221) q[23];
sx q[23];
rz(12.868407135490724) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[11],q[23];
rz(-pi/4) q[23];
cx q[11],q[23];
cx q[11],q[13];
rz(3.664160047912162) q[13];
cx q[11],q[13];
rz(2.4750735916574484) q[11];
id q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[13];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
x q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[1];
cx q[1],q[12];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[13],q[15];
rz(1.4346276272382592) q[13];
rz(pi/2) q[13];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.23106525544074272) q[15];
rz(pi/4) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[22],q[23];
rz(0.21036084021558105) q[23];
cx q[22],q[23];
rz(3.1472732072933) q[22];
rz(pi/2) q[22];
cx q[0],q[22];
rz(0) q[0];
sx q[0];
rz(2.081503828753556) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[22];
sx q[22];
rz(2.081503828753556) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[0],q[22];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(1.2131454612299417) q[0];
rz(-pi/2) q[22];
rz(-3.1472732072933) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/4) q[23];
cx q[23],q[17];
rz(-pi/4) q[17];
cx q[23],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[2],q[17];
rz(-pi/4) q[17];
cx q[17],q[19];
rz(-2.6389429258236925) q[19];
cx q[17],q[19];
rz(1.2558179578032167) q[19];
cx q[2],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-1.3670227433150137) q[16];
sx q[16];
rz(6.516084366537025) q[16];
rz(pi/4) q[16];
rz(pi) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.8014583874615941) q[16];
rz(-0.02526017603913089) q[2];
sx q[2];
rz(5.957297722467791) q[2];
sx q[2];
rz(9.45003813680851) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[20],q[2];
rz(2.9926255802751487) q[2];
cx q[20],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi) q[2];
x q[2];
rz(1.6978399111962579) q[2];
cx q[23],q[6];
cx q[23],q[11];
rz(4.706803778964555) q[11];
cx q[23],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(3.302812745870896) q[11];
rz(pi/2) q[11];
rz(2.722443548232421) q[23];
cx q[23],q[17];
rz(2.548776827155232) q[17];
cx q[23],q[17];
rz(1.667627889361119) q[17];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[3],q[13];
rz(0) q[13];
sx q[13];
rz(1.2965022832851825) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[3];
sx q[3];
rz(1.2965022832851825) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[13];
rz(-pi/2) q[13];
rz(-1.4346276272382592) q[13];
cx q[13],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.2039015347375532) q[13];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[13];
rz(-1.2039015347375532) q[13];
cx q[3],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi) q[13];
x q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[4],q[0];
rz(-3.7219454805607612) q[0];
sx q[0];
rz(1.0103803637653623) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[4],q[0];
rz(0) q[0];
sx q[0];
rz(5.272804943414224) q[0];
sx q[0];
rz(11.9335779801002) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[11];
rz(0) q[0];
sx q[0];
rz(1.5941732820457393) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[11];
sx q[11];
rz(1.5941732820457393) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[0],q[11];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0.9327794771674138) q[0];
cx q[0],q[12];
rz(-pi/2) q[11];
rz(-3.302812745870896) q[11];
rz(-0.9327794771674138) q[12];
cx q[0],q[12];
rz(0.9327794771674138) q[12];
rz(pi/2) q[12];
cx q[23],q[12];
cx q[12],q[23];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.035616195698276) q[12];
rz(0.652942232655735) q[23];
cx q[3],q[0];
rz(0.5183955154111511) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[3],q[16];
rz(-0.8014583874615941) q[16];
cx q[3],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/4) q[4];
cx q[5],q[14];
rz(-pi/4) q[14];
cx q[5],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[15];
rz(-0.23106525544074272) q[15];
cx q[14],q[15];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.326509323319201) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[14];
rz(-2.326509323319201) q[14];
cx q[15],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[15];
rz(-pi/4) q[15];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[4],q[5];
rz(2.01471499262205) q[4];
cx q[4],q[8];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(3.1725218750559048) q[5];
cx q[5],q[19];
rz(-3.1725218750559048) q[19];
sx q[19];
rz(2.0412935539305366) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[5],q[19];
rz(0) q[19];
sx q[19];
rz(4.24189175324905) q[19];
sx q[19];
rz(11.341481878022067) q[19];
rz(pi) q[19];
x q[19];
rz(6.241486001922345) q[5];
rz(2.7927629348190273) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(pi/4) q[10];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[10],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.4052478980717513) q[6];
cx q[6],q[21];
rz(-2.4052478980717513) q[21];
cx q[6],q[21];
rz(2.4052478980717513) q[21];
sx q[6];
rz(-2.01471499262205) q[8];
cx q[4],q[8];
rz(-2.7445681448032095) q[4];
cx q[5],q[4];
rz(-2.7927629348190273) q[4];
sx q[4];
rz(2.400469901325161) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[5],q[4];
rz(0) q[4];
sx q[4];
rz(3.8827154058544253) q[4];
sx q[4];
rz(14.962109040391617) q[4];
cx q[4],q[0];
rz(1.1873240635177413) q[0];
cx q[4],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(4.644835683885333) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(4.748078113953429) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(0.7469517354023676) q[0];
id q[4];
rz(0.6127206198350206) q[4];
sx q[4];
rz(7.0325960751192795) q[4];
rz(-pi/2) q[4];
rz(-0.03609823554847402) q[4];
rz(pi/4) q[5];
cx q[5],q[13];
rz(-pi/4) q[13];
cx q[5],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.473006612240192) q[13];
cx q[13],q[3];
rz(-0.473006612240192) q[3];
cx q[13],q[3];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(0.473006612240192) q[3];
rz(2.01471499262205) q[8];
cx q[9],q[22];
rz(-0.930069526709785) q[22];
cx q[9],q[22];
rz(0.930069526709785) q[22];
cx q[7],q[22];
rz(0.8872882986433579) q[22];
cx q[7],q[22];
rz(3.779489994479179) q[22];
rz(pi/2) q[22];
cx q[22],q[6];
x q[22];
id q[22];
rz(pi) q[22];
x q[22];
rz(pi/2) q[6];
sx q[6];
rz(4.449458214098796) q[6];
sx q[6];
rz(5*pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(0) q[15];
sx q[15];
rz(6.28309813013316) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/4) q[6];
rz(0.6511806028372118) q[6];
rz(pi/4) q[7];
cx q[7],q[1];
rz(-pi/4) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[21],q[1];
cx q[1],q[21];
cx q[21],q[1];
rz(2.0243683948023925) q[1];
sx q[1];
rz(4.135793293157755) q[1];
sx q[1];
rz(9.598255535996977) q[1];
id q[1];
cx q[1],q[15];
rz(0) q[15];
sx q[15];
rz(8.717704642613455e-05) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[1],q[15];
rz(pi/4) q[1];
rz(pi/2) q[1];
rz(pi) q[15];
x q[15];
rz(1.9193563493660823) q[15];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[21];
cx q[20],q[21];
rz(-pi/2) q[20];
cx q[20],q[22];
rz(0.11391375447377466) q[20];
cx q[20],q[3];
rz(-pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
cx q[21],q[19];
rz(3.3340363631892553) q[19];
rz(1.0454973020704654) q[19];
rz(pi/2) q[19];
rz(4.450099622906361) q[21];
sx q[21];
rz(1.6545881990632059) q[21];
cx q[21],q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[21];
rz(0.5031456670933725) q[21];
rz(pi/4) q[22];
rz(-0.11391375447377466) q[3];
cx q[20],q[3];
rz(2.266658437795167) q[20];
rz(0.11391375447377466) q[3];
cx q[15],q[3];
rz(-1.9193563493660823) q[3];
cx q[15],q[3];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(1.9193563493660823) q[3];
rz(0.9162420902811745) q[3];
cx q[3],q[21];
rz(-0.9162420902811745) q[21];
sx q[21];
rz(0.9572199206560228) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[3],q[21];
rz(0) q[21];
sx q[21];
rz(5.325965386523563) q[21];
sx q[21];
rz(9.837874383957182) q[21];
rz(5.436885137510817) q[21];
rz(6.253877586567068) q[3];
rz(-pi/2) q[3];
rz(6.1801001677144765) q[7];
rz(5.392923959497714) q[7];
cx q[2],q[7];
rz(-1.6978399111962579) q[7];
cx q[2],q[7];
cx q[2],q[5];
rz(2.905930740630633) q[2];
sx q[5];
rz(1.6978399111962579) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[18];
rz(0) q[18];
sx q[18];
rz(1.4988708962050525) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[9],q[18];
rz(2.8046453558083435) q[18];
cx q[10],q[18];
rz(-2.8046453558083435) q[18];
cx q[10],q[18];
rz(pi) q[10];
x q[10];
rz(1.1002056092051218) q[10];
rz(2.6017742819636056) q[10];
cx q[10],q[17];
rz(-2.6017742819636056) q[17];
sx q[17];
rz(1.8997652396570204) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[10],q[17];
rz(5.532251881185482) q[10];
sx q[10];
cx q[16],q[10];
rz(-pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[19];
rz(0) q[10];
sx q[10];
rz(0.4616117383529059) q[10];
sx q[10];
rz(3*pi) q[10];
x q[16];
rz(0.7356231157543371) q[16];
rz(4.045944669078376) q[16];
rz(pi/2) q[16];
sx q[16];
rz(5.111111411299191) q[16];
sx q[16];
rz(5*pi/2) q[16];
rz(0) q[16];
sx q[16];
rz(5.033329189132421) q[16];
sx q[16];
rz(3*pi) q[16];
rz(2.4073768866227265) q[16];
rz(pi) q[16];
rz(pi/4) q[16];
rz(0) q[17];
sx q[17];
rz(4.383420067522566) q[17];
sx q[17];
rz(10.358924353371865) q[17];
rz(0.7823251680307777) q[17];
rz(-pi/2) q[18];
cx q[11],q[18];
rz(-0.8998731853102832) q[11];
sx q[11];
rz(6.886334288940064) q[11];
x q[11];
rz(pi) q[11];
rz(0.2186027754730389) q[11];
rz(pi/2) q[18];
rz(5.79742852072677) q[18];
rz(4.797283345786836) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
rz(0) q[19];
sx q[19];
rz(0.4616117383529059) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[10],q[19];
rz(-pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[19];
rz(-1.0454973020704654) q[19];
rz(pi) q[19];
x q[19];
sx q[19];
rz(0.41734466811050663) q[19];
cx q[22],q[11];
rz(3.6668448709879824) q[22];
rz(2.040707933073232) q[9];
sx q[9];
rz(6.941446130185803) q[9];
sx q[9];
rz(13.06890122481188) q[9];
rz(pi/4) q[9];
cx q[9],q[14];
rz(-pi/4) q[14];
cx q[9],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
id q[14];
rz(6.102511142028469) q[14];
rz(2.1559453247622784) q[14];
cx q[14],q[17];
rz(-2.1559453247622784) q[17];
sx q[17];
rz(2.0500365764055832) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[14],q[17];
rz(2.6528173167552693) q[14];
cx q[14],q[12];
rz(-2.6528173167552693) q[12];
cx q[14],q[12];
rz(2.6528173167552693) q[12];
rz(-pi/2) q[12];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(5.211120018217713) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(-pi/2) q[14];
rz(0) q[17];
sx q[17];
rz(4.233148730774003) q[17];
sx q[17];
rz(10.79839811750088) q[17];
cx q[17],q[18];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[17];
cx q[17],q[13];
rz(-pi/4) q[13];
cx q[17],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
id q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(3.7650416825209208) q[18];
rz(-pi/2) q[9];
cx q[8],q[9];
rz(-pi/4) q[8];
sx q[8];
rz(pi/2) q[9];
rz(5.786231121589204) q[9];
rz(5.39353335865715) q[9];
cx q[9],q[23];
rz(-5.39353335865715) q[23];
sx q[23];
rz(0.2187537416534897) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[9],q[23];
rz(0) q[23];
sx q[23];
rz(6.064431565526096) q[23];
sx q[23];
rz(14.165369086770795) q[23];
rz(-3.275596931317676) q[23];
rz(pi/2) q[23];
cx q[7],q[23];
rz(0) q[23];
sx q[23];
rz(2.6240148341933804) q[23];
sx q[23];
rz(3*pi) q[23];
rz(0) q[7];
sx q[7];
rz(3.659170472986206) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[23];
rz(-pi/2) q[23];
rz(3.275596931317676) q[23];
rz(1.8322915481634154) q[23];
rz(0.9689905550028508) q[23];
cx q[23],q[6];
rz(-0.9689905550028508) q[6];
sx q[6];
rz(2.6830326118885903) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[23],q[6];
rz(1.1978445428980535) q[23];
sx q[23];
rz(3.2294553850139045) q[23];
sx q[23];
rz(11.544127043858179) q[23];
cx q[11],q[23];
rz(3.8131095292267765) q[23];
cx q[11],q[23];
rz(0) q[11];
sx q[11];
rz(3.6757066393902624) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[22],q[11];
rz(0) q[11];
sx q[11];
rz(2.607478667789324) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[22],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
sx q[23];
rz(-pi/2) q[23];
cx q[22],q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(7.768437422876888) q[23];
sx q[23];
rz(5*pi/2) q[23];
rz(-pi/2) q[23];
rz(pi/2) q[23];
rz(0) q[6];
sx q[6];
rz(3.600152695290996) q[6];
sx q[6];
rz(9.742587912935019) q[6];
cx q[10],q[6];
rz(2.5395972182398645) q[6];
cx q[10],q[6];
rz(0.8772049327607758) q[6];
cx q[6],q[13];
rz(-0.8772049327607758) q[13];
cx q[6],q[13];
rz(0.8772049327607758) q[13];
cx q[6],q[14];
rz(pi/2) q[14];
rz(5.98704684770771) q[14];
rz(0.8484479032067878) q[14];
cx q[6],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(3.4939093569270288) q[11];
sx q[11];
rz(7.690792633890102) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(3.0552853300995393) q[6];
sx q[6];
rz(4.853918964453995) q[6];
sx q[6];
rz(9.92169244895675) q[6];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.9432112944037843) q[6];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0.8895798972817239) q[7];
cx q[2],q[7];
rz(-2.905930740630633) q[7];
sx q[7];
rz(1.4965170038729396) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(6.154872402645785) q[2];
rz(0) q[7];
sx q[7];
rz(4.786668303306646) q[7];
sx q[7];
rz(11.44112880411829) q[7];
rz(0.020233156093158566) q[7];
sx q[7];
rz(5.997488597235809) q[7];
sx q[7];
rz(14.67042019379489) q[7];
cx q[2],q[7];
rz(4.441684364585469) q[7];
cx q[2],q[7];
rz(3.2900982016940734) q[7];
rz(pi/4) q[7];
x q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
cx q[9],q[8];
rz(pi/2) q[8];
cx q[8],q[5];
cx q[20],q[5];
rz(-2.266658437795167) q[5];
cx q[20],q[5];
rz(2.266658437795167) q[5];
rz(2.4908359856814926) q[5];
cx q[5],q[1];
rz(-2.4908359856814926) q[1];
cx q[5],q[1];
rz(2.4908359856814926) q[1];
rz(0.8023534864578366) q[1];
rz(2.6087428688265693) q[1];
cx q[1],q[19];
rz(-2.6087428688265693) q[19];
sx q[19];
rz(2.3029389278461427) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[1],q[19];
rz(0.8043140915638769) q[1];
cx q[14],q[1];
rz(-0.8484479032067878) q[1];
sx q[1];
rz(0.14100786296652812) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[14],q[1];
rz(0) q[1];
sx q[1];
rz(6.142177444213058) q[1];
sx q[1];
rz(9.46891177241229) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(4.158097119936367) q[14];
sx q[14];
rz(8.147135678439554) q[14];
sx q[14];
rz(15.648202753948553) q[14];
rz(6.034113092658855) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0) q[19];
sx q[19];
rz(3.9802463793334435) q[19];
sx q[19];
rz(11.616176161485441) q[19];
rz(0.014518352958834657) q[19];
rz(-pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[5];
rz(0) q[5];
sx q[5];
rz(3.213616194064983) q[5];
sx q[5];
rz(3*pi) q[5];
x q[8];
id q[8];
cx q[10],q[8];
cx q[8],q[10];
rz(-pi/2) q[10];
cx q[2],q[10];
rz(pi/2) q[10];
cx q[10],q[5];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.035428592930597) q[2];
cx q[21],q[2];
rz(-2.035428592930597) q[2];
cx q[21],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(4.212155110725266) q[2];
rz(pi/4) q[2];
rz(0.06324551282823757) q[2];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0) q[5];
sx q[5];
rz(3.0695691131146035) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[10],q[5];
rz(1.1904638700952566) q[10];
rz(1.5185768167986897) q[10];
rz(1.3633930054501142) q[5];
sx q[5];
rz(5.081175896106908) q[5];
sx q[5];
rz(12.15257215011821) q[5];
cx q[5],q[7];
rz(5.644124813421675) q[5];
rz(0.3136861049660936) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/4) q[7];
rz(2.6503026258926106) q[8];
rz(5.761030184363447) q[8];
sx q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.14585289585094874) q[8];
x q[9];
cx q[9],q[12];
rz(pi/2) q[12];
rz(4.312298561431524) q[12];
cx q[12],q[0];
rz(-4.312298561431524) q[0];
sx q[0];
rz(2.2003633830431095) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[12],q[0];
rz(0) q[0];
sx q[0];
rz(4.082821924136477) q[0];
sx q[0];
rz(12.990124786798535) q[0];
rz(5.956137850244506) q[0];
sx q[0];
rz(8.78476829593621) q[0];
sx q[0];
rz(10.845561122887991) q[0];
rz(pi/2) q[0];
rz(1.3336399854491912) q[12];
rz(1.0582708472532503) q[12];
cx q[12],q[4];
cx q[18],q[9];
rz(-1.0582708472532503) q[4];
sx q[4];
rz(2.8876106923451115) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[12],q[4];
rz(1.1462528096355626) q[12];
rz(0) q[12];
sx q[12];
rz(7.507036316377859) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi) q[12];
rz(pi) q[12];
rz(4.312697352227008) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[4];
sx q[4];
rz(3.3955746148344748) q[4];
sx q[4];
rz(10.519147043571104) q[4];
sx q[4];
cx q[0],q[4];
x q[0];
cx q[4],q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(1.7309365530095062) q[3];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[10],q[1];
rz(-1.5185768167986897) q[1];
cx q[10],q[1];
rz(1.5185768167986897) q[1];
sx q[10];
cx q[23],q[10];
x q[10];
x q[23];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.301524340119065) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[22],q[4];
rz(pi/4) q[22];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[22],q[4];
rz(-pi/4) q[4];
cx q[22],q[4];
rz(0) q[22];
sx q[22];
rz(4.806330004692707) q[22];
sx q[22];
rz(3*pi) q[22];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[22];
rz(0) q[22];
sx q[22];
rz(1.4768553024868794) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[4],q[22];
rz(pi/4) q[22];
cx q[22],q[14];
rz(-pi/4) q[14];
cx q[22],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
x q[22];
cx q[9],q[18];
rz(-pi/2) q[18];
cx q[20],q[18];
cx q[13],q[20];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(3.735546890322846) q[20];
cx q[13],q[20];
cx q[13],q[18];
cx q[18],q[13];
cx q[13],q[18];
sx q[13];
rz(4.257055003974) q[18];
rz(pi/2) q[18];
cx q[19],q[18];
rz(0) q[18];
sx q[18];
rz(1.1936151439871074) q[18];
sx q[18];
rz(3*pi) q[18];
rz(0) q[19];
sx q[19];
rz(1.1936151439871074) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[19],q[18];
rz(-pi/2) q[18];
rz(-4.257055003974) q[18];
sx q[18];
sx q[18];
rz(-pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
cx q[19],q[8];
cx q[20],q[0];
cx q[0],q[20];
rz(3.5006871877858856) q[0];
x q[0];
rz(0.022000301881684083) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[20];
sx q[20];
rz(9.108545032359963) q[20];
sx q[20];
rz(3*pi) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[21],q[20];
rz(1.0563032751966401) q[20];
cx q[21],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[23],q[18];
rz(-0.14585289585094874) q[8];
cx q[19],q[8];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
cx q[4],q[19];
rz(-pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
x q[4];
cx q[5],q[19];
rz(-0.3136861049660936) q[19];
cx q[5],q[19];
rz(0.3136861049660936) q[19];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(5.8888037282701315) q[8];
cx q[1],q[8];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi) q[9];
x q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[17],q[9];
rz(3.37515034671992) q[9];
cx q[17],q[9];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi) q[17];
rz(pi/2) q[17];
cx q[17],q[13];
rz(pi) q[13];
x q[13];
x q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[16],q[17];
rz(-pi/4) q[17];
cx q[16],q[17];
x q[16];
rz(1.627470191103854) q[16];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[6];
cx q[21],q[13];
cx q[13],q[21];
cx q[21],q[13];
rz(pi/2) q[21];
rz(-0.9432112944037843) q[6];
cx q[17],q[6];
rz(pi/4) q[17];
cx q[17],q[1];
rz(-pi/4) q[1];
cx q[17],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[21];
cx q[21],q[6];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[7],q[13];
rz(4.25581183344574) q[13];
cx q[7],q[13];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(0.14837737433961332) q[9];
sx q[9];
rz(4.253625882335362) q[9];
sx q[9];
rz(9.478017154990157) q[9];
cx q[9],q[15];
rz(5.653312351574826) q[15];
cx q[9],q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[20];
rz(5.8827056876993495) q[20];
cx q[15],q[20];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[0];
rz(5.784486299482021) q[0];
cx q[15],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[12];
rz(2.8439152007193584) q[12];
cx q[20],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(2.2679834600138933) q[9];
cx q[9],q[3];
rz(-2.2679834600138933) q[3];
sx q[3];
rz(1.3005182258705175) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[9],q[3];
cx q[11],q[9];
rz(0) q[3];
sx q[3];
rz(4.982667081309069) q[3];
sx q[3];
rz(10.391237080664208) q[3];
rz(0) q[3];
sx q[3];
rz(4.1407962586331895) q[3];
sx q[3];
rz(3*pi) q[3];
x q[3];
cx q[9],q[11];
cx q[16],q[11];
rz(-1.627470191103854) q[11];
cx q[16],q[11];
rz(1.627470191103854) q[11];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[14];
cx q[14],q[9];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
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
