import dbConnect from "@/lib/dbConnect";
import modelSchema from "@/models/modelSchema";

export default async function handler (req, res) {
    await dbConnect();

    const {modelName} = req.query;

    if(req.method === "GET"){
        try {
            const model = await modelSchema.findOne({name: modelName});
            if(!model) return res.status(404).json({message: "Model Not Found!"});
            res.status(200).json(model)
        } catch (error) {
            res.status(500).json({message: "Error Fetching Model", error})
        }
    }
        else{
            res.status(405).json({ message: "Method Not Allowed!"})
        
    }
}