namespace redberry
{
    interface tabTagInterface { }

    public class tabTag : tabTagInterface
    {
        public bool changed = false;
        public string path = null;
        public bool isResult = false;
    }
}