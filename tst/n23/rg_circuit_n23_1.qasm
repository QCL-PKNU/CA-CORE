OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rz(3.723265116488291) q[1];
rz(1.9910559183075989) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.1205777315011463) q[2];
rz(pi/2) q[4];
rz(0) q[5];
sx q[5];
rz(4.481495780068966) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[3],q[5];
rz(0) q[5];
sx q[5];
rz(1.8016895271106204) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[3],q[5];
rz(-3.4124487726447574) q[3];
sx q[3];
rz(4.758908383040961) q[3];
sx q[3];
rz(12.837226733414138) q[3];
rz(2.3100504800867396) q[3];
sx q[3];
rz(8.512788505092) q[3];
sx q[3];
rz(13.016113092831468) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
sx q[6];
cx q[4],q[6];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(8.499592697300457) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[8];
rz(-pi/2) q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi) q[8];
x q[8];
rz(5.015619894400123) q[11];
cx q[11],q[9];
rz(5.811525474957356) q[9];
cx q[11],q[9];
rz(2.2600865603935856) q[11];
rz(2.749198623265124) q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.22807001746886146) q[12];
cx q[1],q[12];
rz(-1.9910559183075989) q[12];
sx q[12];
rz(1.2791562653527868) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[1],q[12];
rz(4.310959832024504) q[1];
rz(0) q[1];
sx q[1];
rz(8.692210732642515) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[12];
sx q[12];
rz(5.004029041826799) q[12];
sx q[12];
rz(11.187763861608117) q[12];
rz(pi/2) q[12];
rz(1.2054974028170315) q[12];
cx q[11],q[12];
rz(-2.749198623265124) q[12];
sx q[12];
rz(2.1824441883648826) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[11],q[12];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[11];
cx q[11],q[1];
rz(-pi/4) q[1];
cx q[11],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi) q[1];
rz(pi/2) q[1];
rz(5.957892048240748) q[11];
rz(pi/2) q[11];
rz(0) q[12];
sx q[12];
rz(4.100741118814704) q[12];
sx q[12];
rz(10.968479181217472) q[12];
id q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.0026362163698638646) q[13];
cx q[0],q[13];
rz(-0.0026362163698638646) q[13];
cx q[0],q[13];
rz(5.1476632403402) q[0];
sx q[0];
rz(4.196668114742051) q[0];
sx q[0];
rz(15.389169778300731) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(1.9513731045665708) q[13];
x q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[2];
rz(-2.1205777315011463) q[2];
cx q[15],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.2816820121882166) q[2];
cx q[8],q[2];
rz(-2.2816820121882166) q[2];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-1.417657751474154) q[17];
rz(pi/2) q[17];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(0.3085794319215873) q[18];
cx q[18],q[5];
rz(-0.3085794319215873) q[5];
cx q[18],q[5];
rz(pi/2) q[18];
rz(0.3085794319215873) q[5];
rz(2.4323437451415377) q[5];
sx q[5];
rz(4.886801088313103) q[5];
sx q[5];
rz(14.726734362287758) q[5];
rz(pi/2) q[19];
sx q[19];
rz(3.5986090274793887) q[19];
sx q[19];
rz(5*pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[20],q[7];
rz(3.9618313909739924) q[7];
cx q[20],q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[14];
rz(0.26658975668834156) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(2.529428437636201) q[14];
sx q[14];
rz(8.313547442650872) q[14];
sx q[14];
rz(15.438693442650461) q[14];
cx q[12],q[14];
rz(5.528929220398058) q[14];
cx q[12],q[14];
sx q[12];
cx q[1],q[12];
x q[1];
rz(1.052820788426906) q[1];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
id q[7];
sx q[7];
cx q[16],q[21];
cx q[21],q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[19],q[16];
rz(1.1172371110592352) q[16];
cx q[19],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(0.1521659399598402) q[16];
sx q[16];
rz(7.83443817551284) q[16];
sx q[16];
rz(14.910608638633466) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[4];
cx q[21],q[10];
rz(pi/2) q[10];
rz(pi/4) q[10];
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
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[0];
rz(-0.6185101151999381) q[10];
sx q[10];
rz(7.363506267942108) q[10];
sx q[10];
rz(10.043288075969318) q[10];
rz(-0.8824644656371594) q[10];
rz(pi/4) q[21];
cx q[21],q[20];
rz(-pi/4) q[20];
cx q[21],q[20];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(pi/4) q[21];
rz(1.4263956185443418) q[4];
cx q[19],q[4];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
cx q[19],q[16];
rz(4.598867168829077) q[16];
cx q[19],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[5];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[8],q[0];
rz(-pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(7.017698418462855) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/4) q[0];
cx q[8],q[5];
rz(3.7298387550626786) q[5];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[8];
rz(-pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[17];
rz(0) q[17];
sx q[17];
rz(1.1726401449271462) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[22];
sx q[22];
rz(5.11054516225244) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[22],q[17];
rz(-pi/2) q[17];
rz(1.417657751474154) q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[6];
rz(-pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
rz(2.875236214798145) q[22];
cx q[15],q[22];
rz(-2.875236214798145) q[22];
cx q[15],q[22];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[21],q[15];
rz(-pi/4) q[15];
cx q[21],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.3610670301777064) q[15];
x q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.6866477730229317) q[21];
rz(1.4701901647382747) q[22];
cx q[22],q[9];
cx q[4],q[21];
rz(-1.6866477730229317) q[21];
cx q[4],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-4.916586366973191) q[21];
rz(pi/2) q[21];
rz(1.9904885804543995) q[4];
rz(2.6918411511530915) q[4];
rz(3.332071640909359) q[6];
cx q[17],q[6];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[17],q[13];
rz(-1.9513731045665708) q[13];
cx q[17],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[3];
rz(pi) q[13];
x q[13];
rz(pi/2) q[13];
sx q[13];
rz(4.776831059616358) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi) q[13];
rz(0.8647566020755487) q[13];
sx q[13];
rz(6.05069606990511) q[13];
sx q[13];
rz(13.569118048358394) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
id q[13];
rz(-0.9656498704267044) q[17];
rz(pi/2) q[17];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(1.8031983909884357) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[3];
sx q[3];
rz(1.8031983909884357) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[11];
rz(-pi/2) q[11];
rz(-5.957892048240748) q[11];
rz(pi/2) q[11];
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
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
cx q[18],q[6];
x q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[17];
rz(0) q[17];
sx q[17];
rz(0.7941354303144754) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[18];
sx q[18];
rz(5.489049876865111) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[17];
rz(-pi/2) q[17];
rz(0.9656498704267044) q[17];
rz(2.0970860132940774) q[17];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(3.820557224035647) q[18];
cx q[18],q[10];
rz(-3.820557224035647) q[10];
sx q[10];
rz(1.5870047790077486) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[18],q[10];
rz(0) q[10];
sx q[10];
rz(4.696180528171838) q[10];
sx q[10];
rz(14.127799650442185) q[10];
rz(5.568344926260348) q[10];
sx q[10];
rz(5.931899511116125) q[10];
sx q[10];
rz(14.908794791013861) q[10];
cx q[10],q[0];
cx q[0],q[10];
rz(0.586648836884669) q[0];
rz(pi/4) q[10];
rz(pi/2) q[10];
rz(-pi/2) q[18];
rz(pi/4) q[18];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[2];
rz(2.660643850058262) q[2];
cx q[6],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[7];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.135186731317616) q[6];
cx q[6],q[15];
rz(-3.135186731317616) q[15];
sx q[15];
rz(1.530900687884062) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[6],q[15];
rz(0) q[15];
sx q[15];
rz(4.752284619295525) q[15];
sx q[15];
rz(11.198897661909289) q[15];
cx q[15],q[5];
rz(-pi/2) q[15];
cx q[11],q[15];
rz(3.943881993890603) q[11];
sx q[11];
rz(8.875485200010479) q[11];
sx q[11];
rz(13.715077558391943) q[11];
rz(pi) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
id q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[21];
rz(0) q[21];
sx q[21];
rz(0.8273414007698832) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[6];
sx q[6];
rz(5.455843906409703) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[21];
rz(-pi/2) q[21];
rz(4.916586366973191) q[21];
rz(5.406629209497767) q[21];
cx q[21],q[0];
rz(-5.406629209497767) q[0];
sx q[0];
rz(0.007581672002612727) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[21],q[0];
rz(0) q[0];
sx q[0];
rz(6.2756036351769735) q[0];
sx q[0];
rz(14.244758333382478) q[0];
rz(0.2686041968067596) q[0];
rz(pi/2) q[0];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[7],q[2];
cx q[2],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[7],q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(-1.052820788426906) q[8];
cx q[1],q[8];
rz(2.8058807271271005) q[1];
rz(1.052820788426906) q[8];
cx q[15],q[8];
rz(-pi/2) q[15];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[0];
rz(0) q[0];
sx q[0];
rz(0.8954128889124973) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[8];
sx q[8];
rz(0.8954128889124973) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[0];
rz(-pi/2) q[0];
rz(-0.2686041968067596) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(2.3683353705947643) q[8];
rz(-1.4701901647382747) q[9];
cx q[22],q[9];
cx q[22],q[20];
rz(pi/2) q[20];
cx q[20],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[3];
rz(-pi/4) q[20];
cx q[20],q[12];
cx q[12],q[20];
rz(-pi/4) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[20],q[6];
rz(pi/2) q[22];
sx q[22];
rz(8.493726308891073) q[22];
sx q[22];
rz(5*pi/2) q[22];
rz(-pi/4) q[22];
rz(pi/2) q[22];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[3];
cx q[22],q[3];
x q[22];
x q[22];
rz(1.0327747740268538) q[3];
rz(pi/2) q[3];
cx q[2],q[3];
rz(0) q[2];
sx q[2];
rz(2.10576348335285) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[3];
sx q[3];
rz(2.10576348335285) q[3];
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
cx q[2],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
x q[2];
rz(4.742411982151679) q[2];
rz(0) q[2];
sx q[2];
rz(6.231267435339483) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[13],q[2];
rz(0) q[2];
sx q[2];
rz(0.05191787184010366) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[13],q[2];
rz(pi) q[13];
rz(pi/2) q[13];
rz(0) q[13];
sx q[13];
rz(7.062636077751961) q[13];
sx q[13];
rz(3*pi) q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(-1.0327747740268538) q[3];
cx q[22],q[3];
rz(0.2888427192258677) q[3];
cx q[22],q[3];
rz(1.877233952465515) q[22];
sx q[22];
rz(8.900062376673326) q[22];
sx q[22];
rz(13.245414845258345) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-0.38960192692721174) q[3];
sx q[3];
rz(9.33707912624324) q[3];
sx q[3];
rz(9.81437988769659) q[3];
rz(1.850301429899129) q[6];
cx q[20],q[6];
rz(-0.7204176428265758) q[20];
sx q[20];
rz(8.25447529467838) q[20];
sx q[20];
rz(10.145195603595955) q[20];
rz(2.0135457618603496) q[20];
sx q[20];
rz(6.56345075740877) q[20];
sx q[20];
rz(11.600339841839897) q[20];
rz(2.4805838567765197) q[20];
rz(pi/2) q[20];
sx q[6];
cx q[10],q[6];
x q[10];
rz(2.672033296026834) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(0.23235861705291327) q[10];
rz(3.129930570380231) q[6];
cx q[7],q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.4701901647382747) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(0) q[9];
sx q[9];
rz(4.353752445087041) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[19],q[9];
rz(0) q[9];
sx q[9];
rz(1.9294328620925456) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[19],q[9];
rz(2.3015891544181404) q[19];
sx q[19];
rz(6.63592774744296) q[19];
sx q[19];
rz(12.380388981244892) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[14],q[19];
rz(4.993209903754711) q[19];
cx q[14],q[19];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(2.9666644014546395) q[19];
cx q[18],q[19];
rz(-2.9666644014546395) q[19];
cx q[18],q[19];
sx q[18];
cx q[16],q[18];
x q[16];
rz(-2.0833556799560875) q[16];
sx q[16];
rz(9.387803532362138) q[16];
sx q[16];
rz(11.508133640725466) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[0];
rz(3.3420527273400307) q[0];
cx q[18],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[18],q[16];
rz(-pi/4) q[16];
cx q[18],q[16];
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
rz(0) q[16];
sx q[16];
rz(3.7088613784668523) q[16];
sx q[16];
rz(3*pi) q[16];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[19],q[15];
rz(pi/2) q[15];
rz(1.448248525849647) q[15];
rz(0.2276534302501092) q[15];
sx q[15];
rz(6.404833468144431) q[15];
rz(0) q[15];
sx q[15];
rz(6.166162087788055) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/2) q[19];
sx q[19];
rz(3.6618322698915566) q[19];
sx q[19];
rz(5*pi/2) q[19];
rz(1.0572144113974538) q[19];
rz(2.367083090014013) q[19];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.4976971215352224) q[7];
sx q[7];
rz(7.350550768533698) q[7];
sx q[7];
rz(10.0514019658079) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(3.5777633260594612) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(3.113848641427769) q[9];
cx q[17],q[9];
rz(-3.113848641427769) q[9];
cx q[17],q[9];
rz(pi/4) q[17];
cx q[17],q[5];
rz(-pi/4) q[5];
cx q[17],q[5];
rz(pi/2) q[17];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
rz(-2.6918411511530915) q[5];
cx q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(7.319117301906606) q[4];
sx q[4];
rz(5*pi/2) q[4];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[4];
cx q[4],q[12];
rz(-pi/4) q[12];
cx q[4],q[12];
cx q[0],q[4];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-0.5179838396311611) q[12];
rz(2.869279729957003) q[4];
cx q[0],q[4];
rz(3.2367194833273856) q[0];
cx q[0],q[16];
rz(0) q[16];
sx q[16];
rz(2.574323928712734) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[0],q[16];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[4];
rz(2.6918411511530915) q[5];
rz(4.757023793814309) q[5];
sx q[5];
rz(8.507393552626844) q[5];
sx q[5];
rz(12.9347007640921) q[5];
rz(pi/2) q[5];
cx q[21],q[5];
cx q[5],q[21];
cx q[21],q[22];
rz(0.7821056698138419) q[22];
cx q[21],q[22];
rz(5.440843951916162) q[21];
cx q[21],q[10];
rz(-5.440843951916162) q[10];
sx q[10];
rz(0.7143103610481769) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[21],q[10];
rz(0) q[10];
sx q[10];
rz(5.568874946131409) q[10];
sx q[10];
rz(14.633263295632627) q[10];
rz(pi/2) q[10];
rz(2.048975789847281) q[21];
sx q[21];
rz(7.574666621321363) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[12];
rz(-3.5777633260594612) q[12];
sx q[12];
rz(1.9468366111483264) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[7],q[12];
rz(0) q[12];
sx q[12];
rz(4.33634869603126) q[12];
sx q[12];
rz(13.52052512646) q[12];
rz(-pi/4) q[12];
rz(3.222857626123303) q[12];
rz(1.540802601970148) q[12];
rz(6.056947793705541) q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
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
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0) q[7];
sx q[7];
rz(5.972189665157645) q[7];
sx q[7];
rz(3*pi) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
sx q[9];
cx q[17],q[9];
x q[17];
rz(pi/4) q[17];
cx q[17],q[14];
rz(-pi/4) q[14];
cx q[17],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
x q[11];
rz(pi/4) q[11];
rz(1.7613175431293762) q[14];
rz(5.612496834195012) q[17];
sx q[17];
rz(5.715771899300373) q[17];
sx q[17];
rz(11.28839224602921) q[17];
rz(-pi/2) q[17];
cx q[5],q[17];
rz(pi/2) q[17];
cx q[17],q[15];
rz(0) q[15];
sx q[15];
rz(0.11702321939153126) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[17],q[15];
rz(2.0323200842596743) q[15];
rz(pi/2) q[15];
rz(pi/2) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[2],q[15];
rz(0) q[15];
sx q[15];
rz(2.17397597298004) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[2];
sx q[2];
rz(2.17397597298004) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[15];
rz(-pi/2) q[15];
rz(-2.0323200842596743) q[15];
rz(pi) q[15];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(5.202301155927343) q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
rz(-pi/4) q[5];
cx q[4],q[5];
rz(0) q[4];
sx q[4];
rz(5.514326888758259) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[20],q[4];
rz(0) q[4];
sx q[4];
rz(0.7688584184213276) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[20],q[4];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0.00032059575366157435) q[20];
rz(pi/2) q[4];
cx q[4],q[15];
rz(2.0062390037545716) q[15];
cx q[4],q[15];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.6217416725494096) q[4];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.227344109217772) q[5];
rz(5.688661662926692) q[5];
cx q[8],q[14];
rz(-1.7613175431293762) q[14];
cx q[8],q[14];
cx q[14],q[7];
rz(0) q[7];
sx q[7];
rz(0.3109956420219415) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[14],q[7];
rz(pi/2) q[14];
sx q[14];
rz(3.15594508137004) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.08718515205636906) q[14];
cx q[21],q[14];
rz(-0.08718515205636906) q[14];
cx q[21],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[22],q[8];
rz(pi/4) q[22];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[22],q[8];
rz(-pi/4) q[8];
cx q[22],q[8];
rz(4.334287734265037) q[22];
cx q[22],q[20];
rz(-0.00032059575366157435) q[20];
cx q[22],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-0.6682828422997376) q[20];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[11],q[8];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(3.716608708439295) q[8];
rz(1.6045666635825322) q[8];
cx q[8],q[5];
rz(-1.6045666635825322) q[5];
cx q[8],q[5];
rz(1.6045666635825322) q[5];
rz(0.8656911745230252) q[8];
sx q[8];
rz(8.16313528748696) q[8];
sx q[8];
rz(14.146503791092073) q[8];
rz(1.8744356678149119) q[8];
sx q[8];
rz(7.456095683614998) q[8];
sx q[8];
rz(15.55121352659651) q[8];
cx q[8],q[12];
rz(4.7572540157731575) q[12];
cx q[8],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(0.49284477722803455) q[9];
cx q[1],q[9];
rz(4.631536241224131) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[19];
rz(-2.367083090014013) q[19];
cx q[1],q[19];
rz(pi/4) q[1];
rz(pi/4) q[1];
cx q[19],q[7];
rz(pi/4) q[19];
rz(3.138704003240744) q[19];
rz(pi/2) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[2];
rz(0) q[2];
sx q[2];
rz(1.7564988617087445) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[7];
sx q[7];
rz(1.7564988617087445) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[2];
rz(-pi/2) q[2];
rz(-5.202301155927343) q[2];
rz(-1.0066817842433964) q[2];
sx q[2];
rz(4.3514302843202435) q[2];
sx q[2];
rz(10.431459745012775) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
id q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[14],q[7];
rz(4.577651035358038) q[7];
cx q[14],q[7];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
rz(-3.129930570380231) q[9];
cx q[6],q[9];
rz(pi) q[6];
rz(pi) q[6];
sx q[6];
cx q[10],q[6];
x q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[1],q[10];
rz(-pi/4) q[10];
cx q[1],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.342639629244225) q[10];
rz(1.8914633217050423) q[10];
sx q[10];
rz(5.471808474800495) q[10];
sx q[10];
rz(12.17382435072457) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[6];
cx q[6],q[22];
rz(pi) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/4) q[6];
cx q[7],q[10];
rz(4.421174384023985) q[10];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(3.129930570380231) q[9];
cx q[3],q[9];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[0];
rz(0.3144726040011043) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[21],q[0];
rz(3.898824355863199) q[0];
cx q[21],q[0];
rz(2.192044572597376) q[0];
cx q[21],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.4353656302644877) q[3];
rz(4.72179121811835) q[3];
cx q[3],q[20];
rz(-4.72179121811835) q[20];
sx q[20];
rz(0.6820322794819145) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[3],q[20];
rz(0) q[20];
sx q[20];
rz(5.601153027697672) q[20];
sx q[20];
rz(14.814852021187466) q[20];
cx q[20],q[4];
rz(0.1492541811885547) q[3];
cx q[3],q[15];
rz(-0.1492541811885547) q[15];
cx q[3],q[15];
rz(0.1492541811885547) q[15];
rz(-pi/4) q[15];
rz(-pi/4) q[15];
rz(pi) q[3];
cx q[3],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[3];
cx q[3],q[16];
rz(-pi/4) q[16];
cx q[3],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.136687024686127) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-0.6217416725494096) q[4];
cx q[20],q[4];
cx q[20],q[22];
rz(0) q[20];
sx q[20];
rz(5.507887827599076) q[20];
sx q[20];
rz(3*pi) q[20];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[4];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
rz(pi/2) q[4];
cx q[16],q[4];
cx q[4],q[16];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(5.235751791984717) q[4];
sx q[4];
rz(3*pi) q[4];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[18];
cx q[9],q[18];
rz(-pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
rz(pi/4) q[18];
cx q[18],q[11];
rz(-pi/4) q[11];
cx q[18],q[11];
rz(pi/4) q[11];
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
cx q[19],q[18];
rz(-3.138704003240744) q[18];
cx q[19],q[18];
rz(3.138704003240744) q[18];
rz(1.114033682084005) q[18];
cx q[18],q[5];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-1.114033682084005) q[5];
cx q[18],q[5];
rz(1.114033682084005) q[5];
rz(0.12564322950805104) q[5];
cx q[14],q[5];
rz(-0.12564322950805104) q[5];
cx q[14],q[5];
rz(6.004932671223917) q[14];
rz(0) q[14];
sx q[14];
rz(8.717632289336814) q[14];
sx q[14];
rz(3*pi) q[14];
sx q[14];
rz(1.9377991756481987) q[5];
cx q[6],q[19];
rz(-pi/4) q[19];
cx q[6],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-0.30403543559966817) q[19];
rz(5.9035875454996045) q[6];
cx q[6],q[19];
rz(-5.9035875454996045) q[19];
sx q[19];
rz(1.9948470989110039) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[6],q[19];
rz(0) q[19];
sx q[19];
rz(4.288338208268582) q[19];
sx q[19];
rz(15.632400941868653) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(2.6104895662079226) q[19];
sx q[19];
rz(5.337933325773517) q[19];
rz(-pi/2) q[19];
rz(pi) q[6];
x q[6];
id q[6];
rz(-0.035396472042244546) q[9];
rz(pi/2) q[9];
cx q[17],q[9];
rz(0) q[17];
sx q[17];
rz(3.8606136006652854) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[9];
sx q[9];
rz(2.422571706514301) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[17],q[9];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(3.0944202889168873) q[17];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[21],q[17];
rz(0.8323858891937982) q[17];
cx q[21],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[17],q[20];
rz(0) q[20];
sx q[20];
rz(0.7752974795805101) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[17],q[20];
rz(1.4210746338538476) q[17];
rz(4.811081930947736) q[20];
rz(1.5416872487393691) q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
cx q[10],q[21];
rz(pi) q[10];
x q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[3],q[20];
rz(0) q[20];
sx q[20];
rz(1.9963677592176534) q[20];
sx q[20];
rz(3*pi) q[20];
rz(0) q[3];
sx q[3];
rz(1.9963677592176534) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[20];
rz(-pi/2) q[20];
rz(-1.5416872487393691) q[20];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[8],q[21];
rz(-pi/4) q[21];
cx q[8],q[21];
rz(pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(4.329510036565932) q[21];
rz(3.9757654731968626) q[8];
rz(-pi/2) q[9];
rz(0.035396472042244546) q[9];
cx q[9],q[1];
cx q[1],q[9];
rz(pi/4) q[1];
cx q[1],q[11];
rz(-pi/4) q[11];
cx q[1],q[11];
cx q[1],q[2];
rz(pi/4) q[1];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[18],q[11];
cx q[11],q[18];
cx q[18],q[11];
cx q[13],q[11];
cx q[11],q[13];
rz(2.8168425886780195) q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(0) q[12];
sx q[12];
rz(0.9662272539973031) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[13];
sx q[13];
rz(0.9662272539973031) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
rz(0.18319796307902908) q[12];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[13];
rz(-2.8168425886780195) q[13];
sx q[18];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
cx q[1],q[22];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
sx q[2];
cx q[22],q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
cx q[15],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(4.086573919957672) q[1];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.37746839279727) q[15];
cx q[13],q[15];
rz(-1.37746839279727) q[15];
cx q[13],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
id q[22];
rz(2.6917152573043026) q[22];
cx q[9],q[0];
rz(-2.192044572597376) q[0];
cx q[9],q[0];
rz(-0.8608301070408193) q[0];
sx q[0];
rz(3.6016840950826308) q[0];
rz(pi/2) q[0];
cx q[0],q[18];
x q[0];
rz(pi/2) q[0];
rz(2.0972429722014065) q[18];
cx q[11],q[18];
rz(-2.0972429722014065) q[18];
cx q[11],q[18];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[7],q[0];
cx q[0],q[7];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[0];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[16],q[7];
rz(pi/4) q[16];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[16],q[7];
rz(-pi/4) q[7];
cx q[16],q[7];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.0702558466872696) q[9];
cx q[5],q[9];
rz(-1.9377991756481987) q[9];
cx q[5],q[9];
rz(0.8279235907555694) q[5];
cx q[17],q[5];
rz(-1.4210746338538476) q[5];
sx q[5];
rz(0.4088765286433098) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[17],q[5];
rz(pi) q[17];
x q[17];
rz(0) q[5];
sx q[5];
rz(5.874308778536276) q[5];
sx q[5];
rz(10.017929003867657) q[5];
rz(1.4018733788419162) q[5];
sx q[5];
rz(5.398697107759932) q[5];
sx q[5];
rz(13.612692349401863) q[5];
rz(1.9377991756481987) q[9];
rz(pi/2) q[9];
cx q[9],q[2];
rz(0) q[2];
sx q[2];
rz(4.190646965510966) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[6],q[2];
rz(0) q[2];
sx q[2];
rz(2.0925383416686194) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[6],q[2];
x q[9];
cx q[9],q[19];
rz(pi/2) q[19];
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