import  pandas as pd
import os, sys

if __name__ == '__main__':
    PRJ_NAME = ["Key_detect_171220",
                "CM_171219"
               ]
    PRJ_NAME = ["Key_detect_171220"]
    CONFUSION_MATRIX =os.path.join(os.environ['KV_CACHE'],'confusion')
    for prj_name in PRJ_NAME:
        CONFUSION_DIR = os.path.join(os.environ['KV_DATA'],'confusion',prj_name)
    # Get all xls file
        xls_list = []
        for file in sorted(os.listdir(CONFUSION_DIR)):
            if file.endswith(".xlsx") and not file.startswith("~"):
                xls_list.append(file)
        print xls_list
        frames=[]
        for file in xls_list:
            df = pd.read_excel(CONFUSION_DIR + "/" + file)
            df2 = df.copy()
            print df2.head()
            # drop unwanted colum
            df2 = df2.drop(df2.columns[0],axis=1)
            df2 = df2.drop(df2.columns[3],axis=1)
            #remove na cel
            df2 = df2.dropna()
            frames.append(df2)

        df_out= pd.concat(frames)
        df_out.index.names = ['index']
        df_out = df_out.rename(index=str,columns={'Image Name':'id','OCR value':'value','Right Answer':'truth'})
        print df_out.head()

        df_out.to_csv(CONFUSION_MATRIX +'/'+  prj_name+'.csv',sep=',',encoding='utf-8')

