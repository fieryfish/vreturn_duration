import java.io.IOException;
import java.util.Arrays;

public class CallPy {

    public static void main(String[] args) throws IOException {
        String[] command = new String[2];
        command[0]="python";
        command[1]=System.getProperty("user.dir")+"/production.py"; //path to the script
        String runPath = "python /Users/yulong/Documents/code/workspace/oih/vreturn_duration/production.py";
        System.out.println(runPath);
        Runtime.getRuntime().exec(runPath);

        //command[2]="-F"; //argument/Flag/option
        //command[3]="--dir="+path; //argument/option
        //command[4]="--filename="+filename; //argument/option 
        //command[5]=argument; //argument/option
    }
}
