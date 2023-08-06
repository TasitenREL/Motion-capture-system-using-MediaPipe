using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;//Listを使うために追加
using System.IO;//テキスト書き込みのために追加
using UnityEngine;
using UnityEngine.Networking;//UnityWebRequestを使うために追加
using System.Text;
using Valve.VR;
using Newtonsoft.Json;//Jsonデータを扱うために追加
using Newtonsoft.Json.Linq;//Jsonデータを扱うために追加

public class humanPos6 : MonoBehaviour
{
    void Update()
    {
        StartCoroutine(Post("hecate.local:50505", "{\"getval\":\"human_pose\"}"));//コルーチンで通信処理
    }

    IEnumerator Post(string url, string bodyJsonString)//Post通信
    {
        using var request = new UnityWebRequest(url, "POST");//UnityWebRequestを生成、usingをつけてメモリを解放している。
        byte[] bodyRaw = Encoding.UTF8.GetBytes(bodyJsonString);//文字列からバイト型配列に変換
        request.uploadHandler = (UploadHandler)new UploadHandlerRaw(bodyRaw);//サーバーへ送信するデータを扱うオブジェクト
        request.downloadHandler = (DownloadHandler)new DownloadHandlerBuffer();//サーバーから受信するデータを扱うオブジェクト
        request.SetRequestHeader("Content-Type", "application/json");//リクエストヘッダの設定
        yield return request.Send();//リクエスト送信

        var obj = JsonConvert.DeserializeObject<Dictionary<string, object>>(request.downloadHandler.text);//Jsonから辞書型に変換
        var Return = (JObject)obj["RETURN"];
        var OpnePose = (JObject)Return["openpose"];
        
        for (int p = 0; p < OpnePose.Count; ++p)//人の数だけループ
        {
            Debug.Log("person" + (p + 1));
            var person = (JObject)OpnePose["person" + (p + 1)];
            
            for (int i = 0; i < person.Count; ++i)//関節の数だけループ
            {
                try
                {
                    var m = (JArray)person["m" + i];
                    string s = "human" + (p + 1).ToString() + "/" + m[1].ToString();//関節番号m[1]をstr型にキャストし、s変数に格納
                    GameObject trackerob = GameObject.Find(s);//オブジェクトを見つける
                    Vector3 Opos = trackerob.transform.position;
                    Opos.x = (float)m[2] * -2;
                    Opos.y = (float)m[3] * -2;
                    Opos.z = (float)m[4] * -2;
                    trackerob.transform.position = Opos;
                }
                catch (System.NullReferenceException e)
                {
                    Debug.Log("error");
                }
            }
        }
    }
}
