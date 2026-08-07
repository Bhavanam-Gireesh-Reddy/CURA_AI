function validateField(id, min, max) {

    const input = document.getElementById(id);
    const error = document.getElementById(id + "_error");

    const value = parseFloat(input.value);

    if (input.value === "") {

        error.innerHTML = "This field is required";
        input.classList.add("invalid");
        return false;
    }

    if (isNaN(value)) {

        error.innerHTML = "Enter a valid number";
        input.classList.add("invalid");
        return false;
    }

    if (value < min || value > max) {

        error.innerHTML = `Enter value between ${min} and ${max}`;
        input.classList.add("invalid");
        return false;
    }

    error.innerHTML = "";
    input.classList.remove("invalid");

    return true;
}



function validateForm() {

    let valid = true;

    valid &= validateField("Age", 1, 120);

    valid &= validateField("Total_Bilirubin", 0.1, 1.5);

    valid &= validateField("Direct_Bilirubin", 0.0, 1);

    valid &= validateField("Alkaline_Phosphotase", 10, 200);

    valid &= validateField("Alamine_Aminotransferase", 1, 86);

    valid &= validateField("Aspartate_Aminotransferase", 1, 57);

    valid &= validateField("Total_Protiens", 2, 10);

    valid &= validateField("Albumin", 0.5, 6);

    valid &= validateField("Albumin_and_Globulin_Ratio", 0.1, 3);

    document.getElementById("predictBtn").disabled = !valid;

    return valid;
}



document.getElementById("liverForm")
.addEventListener("submit", async function (e) {

    e.preventDefault();

    if (!validateForm()) {
        return;
    }

    const data = {

        model: document.getElementById("model").value,

        Age: Number(document.getElementById("Age").value),

        Gender: Number(document.querySelector("[name='Gender']").value),

        Total_Bilirubin: Number(document.getElementById("Total_Bilirubin").value),

        Direct_Bilirubin: Number(document.getElementById("Direct_Bilirubin").value),

        Alkaline_Phosphotase: Number(document.getElementById("Alkaline_Phosphotase").value),

        Alamine_Aminotransferase: Number(document.getElementById("Alamine_Aminotransferase").value),

        Aspartate_Aminotransferase: Number(document.getElementById("Aspartate_Aminotransferase").value),

        Total_Protiens: Number(document.getElementById("Total_Protiens").value),

        Albumin: Number(document.getElementById("Albumin").value),

        Albumin_and_Globulin_Ratio: Number(
            document.getElementById("Albumin_and_Globulin_Ratio").value
        )

    };

    const response = await fetch("/liver_predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(data)

    });

    const result = await response.json();

    document.getElementById("liverResult").innerHTML = `

    <div class="result">

        <div class="prediction-box">

            <h3>${result.prediction}</h3>

            <p><strong>Selected Model:</strong> ${result.model_used}</p>

            <p><strong>Confidence:</strong> ${result.confidence}%</p>

        </div>

    </div>

    `;
});
// =========================
// CURA AI RAG CHATBOT
// =========================

document.addEventListener("DOMContentLoaded", function () {

    const sendButton = document.getElementById("sendChat");
    const input = document.getElementById("chatQuestion");
    const messages = document.getElementById("chatMessages");


    sendButton.addEventListener("click", async function () {


        let question = input.value.trim();


        if (question === "")
            return;



        messages.innerHTML += `

        <div class="user-message">
            ${question}
        </div>

        `;


        input.value = "";



        let loading = document.createElement("div");

        loading.className = "bot-message";

        loading.innerHTML = "Thinking...";


        messages.appendChild(loading);



        try {


            const response = await fetch("/rag_query", {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    question: question

                })

            });



            const data = await response.json();



            if (data.answer) {


                let answer = data.answer;



                // Convert markdown to HTML

                answer = answer
                    .replace(
                        /\*\*(.*?)\*\*/g,
                        "<strong>$1</strong>"
                    )
                    .replace(
                        /\n\n/g,
                        "<br><br>"
                    )
                    .replace(
                        /\n/g,
                        "<br>"
                    );



                let button = "";


if(data.tool){


if(data.tool.tool_name==="liver"){

button = `

<button onclick="location.href='#liver'">

🩺 Check Liver Disease

</button>

`;

}



else if(data.tool.tool_name==="xray"){


button = `

<button onclick="location.href='#xray'">

🦴 Detect Fracture

</button>

`;

}



else if(data.tool.tool_name==="mri"){


button = `

<button onclick="location.href='#mri'">

🧠 Analyze MRI

</button>

`;

}


}



loading.innerHTML = `

<div class="rag-response">

${answer}

<br><br>

${button}

</div>

`;


            }

            else {


                loading.innerHTML = `

                <div class="rag-response error">

                    ${data.error || "No response generated"}

                </div>

                `;


            }



        }


        catch (error) {


            console.log(error);


            loading.innerHTML = `

            <div class="rag-response error">

                Server connection failed.

            </div>

            `;


        }


    });



    // Enter key support

    input.addEventListener(
        "keypress",
        function (e) {

            if (e.key === "Enter") {

                sendButton.click();

            }

        }
    );


});